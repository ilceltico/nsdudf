
from core.utils import get_query_points, df_and_grad_to_input_cells
import core.utils as utils
import sys
sys.path.append("custom_mc")
from _marching_cubes_lewiner import pseudosdf_mc_lewiner
#Import DualMesh-UDF if present, otherwise skip
try:
    from DualMeshUDF.extract_mesh import extract_mesh_mod, extract_mesh
except:
    print("DualMesh-UDF not found. Skipping import.")
    pass
import time

import torch
import numpy as np
import trimesh

POWERS_OF_2 = utils.POWERS_OF_2


def compute_pseudo_sdf(model, udf_and_grad_f, n_grid_samples=128, batch_size=10000, normalize_udf=True, use_grads=True, out7=False):
    """Computes the pseudo-sdf of a mesh using a neural network model and a UDF and gradient function.
    
    Args:
        model: The neural network model.
        udf_and_grad_f: A function that takes query points and returns the UDF and gradients.
        n_grid_samples: The number of grid samples in each dimension.
        batch_size: The batch size to use when computing the pseudo-sdf. You can play with this if you run of out memory.
        normalize_udf: Whether to normalize the UDF values with the voxel size. This depends on how the model was trained. Change this only if you train the model with a different setting for this parameter.
        use_grads: Whether to use gradients in the pseudo-sdf computation. This depends on how the model was trained. Change this only if you train the model with a different setting for this parameter.
        out7: Whether the model has 7 outputs (compared to the default 128). This depends on how the model was trained. Change this only if you train the model with a different setting for this parameter.
    """
    device = next(model.parameters()).device

    bbox = [(-1., -1., -1.), (1., 1., 1.)]
    query_points = get_query_points(bbox, n_grid_samples).view(-1,3)

    print(f"Extracting UDF and gradients...", end=" ", flush=True)
    start = time.time()
    udf, grads = udf_and_grad_f(query_points)
    print(f"Done in: {time.time() - start} seconds")

    print("Computing Pseudo-SDF...", end=" ", flush=True)
    start = time.time()

    udf = udf.view(n_grid_samples,n_grid_samples,n_grid_samples)
    grads = grads.view(n_grid_samples,n_grid_samples,n_grid_samples,3)
    
    shape = (n_grid_samples - 1, n_grid_samples - 1, n_grid_samples - 1)

    voxel_size = 2.0 / (n_grid_samples - 1)
    # Limits the meshing distance using the cell's average and maximum distances. 
    # This speeds up the computation and reduces the number of unwanted triangles.
    # The same thresholds are used in the training of the network.
    max_avg_distance = 1.2 * voxel_size
    max_max_distance = 2.0 * voxel_size	
    udf_cells, grad_cells, indices = df_and_grad_to_input_cells(udf, grads, max_avg_distance, max_max_distance)

    with torch.no_grad():

        # Default value for the pseudo-sdf is 1.0. This ensures that cells that are not near the mesh are not meshed, since their SDF is non-zero.
        pseudo_sdf = torch.ones(shape + (8,))

        cell_size = 32 if use_grads else 8

        # Build the input by putting the cells together
        input = torch.zeros((len(indices), cell_size))
        input[:,:8] = udf_cells[:,indices].T
        if use_grads:
            input[:,8:] = grad_cells[:,indices,:].permute(1,0,2).reshape(-1,24)


        udf2 = input[:,:8].clone()
        udf2 = udf2.to(device)

        # Normalize the UDF values with the voxel size. This is key for the network to work at multiple resolutions.
        if normalize_udf:
            input[:,:8] = input[:,:8] / voxel_size

        input = input.to(device)

        output = torch.zeros((input.shape[0],7 if out7 else 128)).to(device)
        batch_head = 0
        while batch_head < input.shape[0]:
            batch_tail = min(batch_head + batch_size, input.shape[0])
            output[batch_head:batch_tail] = model(input[batch_head:batch_tail])
            batch_head += batch_size

        # Compute the pseudo-signs based on the network's output configuration
        if out7:
            pseudo_signs = (torch.sigmoid(output) > 0.5).int()
        else:
            pseudo_signs = output.argmax(axis=1).unsqueeze(-1).bitwise_and(POWERS_OF_2.to(device)).ne(0).int()	
        pseudo_signs[pseudo_signs == 0] = -1

        # Multiply the pseudo-signs with the udf values to get a pseudo-sdf
        pseudo_sdf[np.unravel_index(indices, shape)] = torch.hstack((udf2[:,0].reshape(-1,1), udf2[:,1:8] * pseudo_signs)).cpu()

    print(f"Done in: {time.time() - start} seconds")

    return pseudo_sdf.detach().numpy()



def mesh_marching_cubes(pseudo_sdf):
    """Extracts a mesh from a pseudo-sdf using the marching cubes algorithm."""

    print("Extracting mesh using Marching Cubes...", end=" ", flush=True)
    start = time.time()

    resolution = pseudo_sdf.shape[0] + 1
    voxel_size = 2.0 / (resolution - 1)
    try:
        vertices, faces, normals, values = pseudosdf_mc_lewiner(pseudo_sdf, spacing=[voxel_size] * 3)
    except:
        print(f"Failed to mesh")
        return None
    vertices = vertices - 1 # Since voxel origin is [-1,-1,-1]
    mesh = trimesh.Trimesh(vertices, faces)

    print(f"Done in: {time.time() - start} seconds")

    return mesh



def mesh_dual_mesh_udf(pseudo_sdf, udf_f_dmudf, udf_grad_f_dmudf, batch_size=10000, device="cpu"):
    depth = int(np.ceil(np.log2(pseudo_sdf.shape[0])))
    grid_points = pseudo_sdf.shape[0] + 1
    if (np.log2(pseudo_sdf.shape[0]) != depth):
        raise ValueError("The pseudo-sdf must have a resolution that is a power of 2. This amounts to a grid resolution of 2^depth + 1. Try 129 or 257.")

    # Extract the mesh using the normal DualMesh-UDF, untuned.
    mesh_v_orig, mesh_f_orig, _, _, _ = extract_mesh(udf_f_dmudf, udf_grad_f_dmudf, batch_size=batch_size, max_depth=depth)

    # Now we extract the mesh using a "relaxed" version of DualMesh-UDF, to make sure we include as many faces as possible.
    mesh_v, mesh_f, _, _, _ = extract_mesh_mod(udf_f_dmudf, udf_grad_f_dmudf, batch_size=batch_size, max_depth=depth)


    torch_mesh_v = torch.Tensor(mesh_v).to(device)
    #Get which cell each vertex belongs to
    #TODO, for simplicity I'm not filtering vertices outside of the grid
    #In fact, the spurious vertices we've added to the mesh cannot go out the grid because they are taken as cell centers, and those are the ones that the network should filter anyway
    cell_indices = torch.floor(torch_mesh_v / (2.0 / (grid_points - 1)) + (grid_points-1)/2).int().to('cpu')

    #To do so, set out of bounds cell_indices as grid_points,grid_points,grid_points
    cell_indices[cell_indices < 0] = grid_points-1
    cell_indices[cell_indices >= grid_points-1] = grid_points-1

    # Define additional cells for the pseudo-SDF, out of the normal bounds, with additional 0 values for grid_points,grid_points,grid_points
    # So that these vertices are not filtered out
    current_pseudo_sdf = torch.zeros((grid_points,grid_points,grid_points,8))
    current_pseudo_sdf[:-1,:-1,:-1] = torch.Tensor(pseudo_sdf).to(device)

    #Get the pseudo-SDF for each cell index
    cell_pseudo_sdf = current_pseudo_sdf[tuple(cell_indices.T)].unsqueeze(1).reshape(-1,8)

    #Filter out faces that contain a vertex whose cell has a pseuso-SDF with no negative values (i.e. there are no predicted sign flips)
    filtered_mesh_f_neural = mesh_f[torch.all(torch.any(cell_pseudo_sdf[mesh_f] <= 0, axis=2), axis=1)]
    
    # Merge the original and the neural meshes in trimesh
    full_mesh_v = np.vstack((mesh_v, mesh_v_orig))
    full_mesh_f = np.vstack((filtered_mesh_f_neural, mesh_f_orig + mesh_v.shape[0]))

    pseudosdf_dmudf_mesh = trimesh.Trimesh(full_mesh_v, full_mesh_f)
    #Remove duplicated faces
    pseudosdf_dmudf_mesh.remove_duplicate_faces()

    #Retrieve also the original DualMesh-UDF mesh
    dmudf_mesh = trimesh.Trimesh(mesh_v_orig, mesh_f_orig)

    return dmudf_mesh, pseudosdf_dmudf_mesh
