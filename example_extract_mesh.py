import trimesh
import igl
import numpy as np
import torch
from trimesh.transformations import scale_matrix

import utils
from meshing import compute_pseudo_sdf, mesh_marching_cubes



device = "cpu"
resolution = 256


def main():
    # Load the model
    model = utils.load_model("model.pt", device)

    # Now we need to define a function to extract the UDF and gradients
    # Here we use a UDF computed from an ABC example object (validation set). It is object 00009484, the same shown in the paper.

    # We load the object
    gt_mesh = trimesh.load("00009484.obj")

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

    # We can now extract the mesh
    pseudo_sdf = compute_pseudo_sdf(model, lambda query_points: udf_and_grad_f(query_points, gt_mesh), n_grid_samples=resolution)
    mesh = mesh_marching_cubes(pseudo_sdf)
    
    # De-normalize the mesh to the original size
    mesh = mesh.apply_transform(scale_matrix(max(gt_mesh_extents) / 1.99999))
    mesh = mesh.apply_translation(gt_mesh_bounds)
    mesh.export(f"extracted_mesh_{resolution}.obj")

    # The mesh can be postprocessed to fill small cracks, holes, smooth the surface, remove degenerate faces, etc.
    # Results in the paper do not include any postprocessing.

    # Example of simple postprocessing
    mesh.fill_holes()
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, validate=True)
    mesh.export(f"extracted_mesh_{resolution}_postprocessed.obj")




# Define a function that extracts UDF and gradients and returns them as Torch Tensors. 
def udf_and_grad_f(query_points, mesh):
    udf, facet_indices, closest_points = igl.point_mesh_squared_distance(query_points.cpu().detach().numpy(), mesh.vertices, mesh.faces) #This function computes the squared distance, so we need to take the square root
    udf = np.sqrt(udf)
    udf = torch.Tensor(udf)

    # IMPORTANT: the gradients point away from the surface.
    udf_grads = query_points - closest_points
    udf_grads = torch.Tensor(udf_grads)
    udf_grads_normalized = udf_grads / torch.linalg.norm(udf_grads, axis=1).reshape(-1,1)

    # Some query points are exactly on the surface and can produce NaN gradients
    # The UDF gradient does not exist on the surface, so here we set it to zero.
    udf_grads_normalized = torch.nan_to_num(udf_grads_normalized, nan=0.0)

    return udf, udf_grads_normalized


# Same as above, but for neural UDFs.
# For speed purposes, batching is IMPORTANT.
import torch.nn.functional as F
def udf_and_grad_neural_f(model, latent, query_points, max_batch=32**3):
    model.eval()

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

        udf_sub = model(inputs)
        udf[i : i + max_batch] = udf_sub.squeeze(1).detach().cpu()
        udf_sub.sum().backward(retain_graph=True)
        angle = xyz_subset.grad.detach()
        # IMPORTANT: the gradients point away from the surface.
        grad[i : i + max_batch,:] = F.normalize(angle, dim=1)
        
    return udf, grad




if __name__ == "__main__":
    main()