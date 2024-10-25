import trimesh
import igl
import numpy as np
import torch
from trimesh.transformations import scale_matrix
import argparse

import core.utils as utils
from core.meshing import compute_pseudo_sdf, mesh_marching_cubes, mesh_dual_mesh_udf



def main():
    #Parse arguments from command line
    parser = argparse.ArgumentParser(description='Extract a mesh from a UDF.')
    parser.add_argument('--resolution', type=int, default=257, help='Number of grid samples per axies used to extract the mesh.')
    parser.add_argument('--batch_size', type=int, default=10000, help='Batch size for computing the UDF and gradients.')
    parser.add_argument('--device', type=str, default="cuda", help='Device to use for computing the UDF and gradients.')
    parser.add_argument('--model', type=str, default="model.pt", help='Path to the model file.')
    parser.add_argument('--object', type=str, default="example_objects/shapenet_cars_10247b51a42b41603ffe0e5069bf1eb5.obj", help='Path to the object file.')
    parser.add_argument('--meshing_algo', type=str, default="marching_cubes", help='Meshing algorithm to use. Options are "marching_cubes" and "dual_mesh_udf".')

    args = parser.parse_args()
    # pretty print the arguments
    print(args)

    # If CUDA is not available, use CPU
    if not torch.cuda.is_available():
        print("CUDA not available. Using CPU.")
        args.device = "cpu"

    # Load the model
    model = utils.load_model(args.model, args.device)

    # Now we need to define a function to extract the UDF and gradients
    # Here we use a UDF computed example object, by default it is a car from ShapeNet, the same used in the teaser.
    # You can use any other object, take a look at the `example_objects` folder for more examples.

    # We load the object
    gt_mesh = trimesh.load(args.object)

    # We remove non-mesh information
    if isinstance(gt_mesh, trimesh.Scene):
        if len(gt_mesh.geometry) == 0:
            gt_mesh = None  # empty scene
        else:
            # we lose texture information here
            gt_mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in gt_mesh.geometry.values()))

    # We normalize it to a [-1,1] bounding box. 
    # This normalization is not strictly needed but it helps with meshing. We can de-normalize the mesh later.
    # Try to mesh the object without normalization. You should get similar results. 
    # So, if normalizing is not possible (e.g. a neural UDF that has been trained without normalization), it is not a problem.
    gt_mesh_bounds = gt_mesh.bounds.mean(axis=0)
    gt_mesh_extents = gt_mesh.extents
    gt_mesh.apply_translation(-gt_mesh_bounds)
    scale = scale_matrix(1.99999 / max(gt_mesh_extents)) #Keeping 2.0 causes some faces to be exactly on the edge of the box, which is potentially problematic
    gt_mesh.apply_transform(scale)


    # Pseudo-SDF computation
    pseudo_sdf = compute_pseudo_sdf(model, lambda query_points: udf_and_grad_f(query_points, gt_mesh), n_grid_samples=args.resolution, batch_size=args.batch_size)


    # We can now extract the mesh

    if args.meshing_algo == "marching_cubes":
        mesh = mesh_marching_cubes(pseudo_sdf)
        
        # De-normalize the mesh to the original size, if needed/possible
        mesh = mesh.apply_transform(scale_matrix(max(gt_mesh_extents) / 1.99999))
        mesh = mesh.apply_translation(gt_mesh_bounds)

        mesh.export(f"extracted_mesh_{args.resolution}.obj")

        # The mesh can be postprocessed to fill small cracks, holes, smooth the surface, remove degenerate faces, etc.
        # Results in the paper do not include any postprocessing.

        # Example of simple postprocessing
        print("Postprocessing mesh...")
        mesh.fill_holes()
        mesh.update_faces(mesh.unique_faces())
        mesh.remove_unreferenced_vertices()
        mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, validate=True)
        mesh.export(f"extracted_mesh_{args.resolution}_postprocessed.obj")

    if args.meshing_algo == "dual_mesh_udf":
        # We can also extract the mesh using a modified version of DualMesh-UDF. We refer to the paper for more details.
        # In short: we relax the parameters of DualMesh-UDF to allow for more triangles in the mesh, and filter out unwanted ones using our Pseudo-SDF.
        # We also add triangles in cells where DualMesh-UDF fails but the Pseudo-SDF predicts a surface.
        print("Extracting mesh using DualMesh-UDF...")
        dmudf_mesh, pseudosdf_dmudf_mesh = mesh_dual_mesh_udf(pseudo_sdf, lambda query_points: udf_f_dmudf(query_points, gt_mesh), lambda query_points: udf_grad_f_dmudf(query_points, gt_mesh), batch_size=args.batch_size, device=args.device)
        
        pseudosdf_dmudf_mesh = pseudosdf_dmudf_mesh.apply_transform(scale_matrix(max(gt_mesh_extents) / 1.99999))
        pseudosdf_dmudf_mesh = pseudosdf_dmudf_mesh.apply_translation(gt_mesh_bounds)
        dmudf_mesh = dmudf_mesh.apply_transform(scale_matrix(max(gt_mesh_extents) / 1.99999))
        dmudf_mesh = dmudf_mesh.apply_translation(gt_mesh_bounds)
        pseudosdf_dmudf_mesh.export(f"extracted_mesh_ours+dmudf_{args.resolution}.obj")
        dmudf_mesh.export(f"extracted_mesh_dmudf_{args.resolution}.obj")



# Define a function that extracts UDF and gradients and returns them as Torch Tensors. 
def udf_and_grad_f(query_points, mesh):
    udf, facet_indices, closest_points = igl.point_mesh_squared_distance(query_points.cpu().detach().numpy(), mesh.vertices, mesh.faces) #This function computes the squared distance, so we need to take the square root
    udf = np.sqrt(udf)
    udf = torch.Tensor(udf)

    # IMPORTANT: the gradients point away from the surface.
    udf_grads = query_points - closest_points
    udf_grads = torch.Tensor(udf_grads)
    udf_grads_normalized = utils.normalize(udf_grads, dim=1)

    # Some query points are exactly on the surface and can produce NaN gradients
    # The UDF gradient does not exist on the surface, so here we set it to zero.
    udf_grads_normalized = torch.nan_to_num(udf_grads_normalized, nan=0.0)

    return udf, udf_grads_normalized


# Here is an example of the above function, but for neural UDFs (udf_autodecoder in this example)
# For speed purposes, batching is IMPORTANT.
import torch.nn.functional as F
def udf_and_grad_f2(udf_autodecoder, latent, query_points, max_batch=32**3):
    udf_autodecoder.eval()

    # Prepare data
    xyz_all = query_points.view(-1, 3)
    #xyz_all.requires_grad = True
    n_points = len(xyz_all)
    udf = torch.zeros(n_points)
    grad = torch.zeros(n_points,3)

    # Predict UDF on a subset of points at a time
    latent_rep = latent.expand(max_batch, -1)
    for i in range(0, n_points, max_batch):
        xyz_subset = xyz_all[i : i + max_batch].cuda()
        xyz_subset.requires_grad = True
        inputs = torch.cat([latent_rep[:len(xyz_subset)], xyz_subset], dim=-1)

        udf_sub = udf_autodecoder(inputs)
        udf[i : i + max_batch] = udf_sub.squeeze(1).detach()
        udf_sub.sum().backward(retain_graph=True)
        angle = xyz_subset.grad.detach()
        # IMPORTANT: the gradients point away from the surface.
        grad[i : i + max_batch,:] = F.normalize(angle, dim=1)
        
    return udf, grad



###################################
########## DUALMESH-UDF ###########
###################################

# The following functions are examples used in the DualMesh-UDF code.
# They are similar to the ones above, but they return the results in sligthly different formats.
# Implement your own functions for your UDFs.
def udf_f_dmudf(query_points, mesh):
    return np.sqrt(igl.point_mesh_squared_distance(query_points, mesh.vertices, mesh.faces)[0]).reshape(-1,1)

def udf_grad_f_dmudf(query_points, mesh):
    udf, facet_indices, closest_points = igl.point_mesh_squared_distance(query_points, mesh.vertices, mesh.faces)
    udf = np.sqrt(udf)

    udf_grads = query_points - closest_points
    udf_grads = torch.Tensor(udf_grads)
    # udf_grads_normalized = udf_grads / torch.linalg.norm(udf_grads, axis=1).reshape(-1,1)
    udf_grads_normalized = utils.normalize(udf_grads, dim=1).reshape(-1,1)
    return udf.reshape(-1,1), udf_grads_normalized.reshape(-1,3,1)

# Here is also an example of the above functions, but for neural UDFs.
def udf_f_autodecoder_dmudf(net, latent_vec, device):
    def udf(pts, net=net, device=device):
        net.eval()
        target_shape = list(pts.shape)

        pts = pts.reshape(-1, 3)
        xyz = torch.from_numpy(pts).to(device)
        
        batch_vecs = latent_vec.view(latent_vec.shape[0], 1, latent_vec.shape[1]).repeat(1, target_shape[0], 1)
        input = torch.cat([batch_vecs.reshape(-1, latent_vec.shape[1]), xyz.reshape(-1, xyz.shape[-1]).float()], dim=1)
        udf_p = net(input)
        
        target_shape[-1] = 1
        udf_p = udf_p.reshape(target_shape).detach().cpu().numpy()

        return udf_p
    return udf

def udf_grad_f_autodecoder_dmudf(net, latent_vec, device):
    def udf_grad(pts, net=net, device=device):
        net.eval()
        target_shape = list(pts.shape)

        pts = pts.reshape(-1, 3)
        xyz = torch.from_numpy(pts).to(device)
        pts.requires_grad = True

        batch_vecs = latent_vec.view(latent_vec.shape[0], 1, latent_vec.shape[1]).repeat(1, target_shape[0], 1)
        input = torch.cat([batch_vecs.reshape(-1, latent_vec.shape[1]), xyz.reshape(-1, xyz.shape[-1]).float()], dim=1)
        udf_p = net(input)

        udf_p.sum().backward()
        grad_p = pts.grad.detach()
        grad_p = utils.normalize(grad_p)

        grad_p = grad_p.reshape(target_shape).detach().cpu().numpy()
        target_shape[-1] = 1
        udf_p = udf_p.reshape(target_shape).detach().cpu().numpy()

        return udf_p, grad_p
    return udf_grad


if __name__ == "__main__":
    main()