
from utils import get_query_points
import utils
import sys
sys.path.append("custom_mc")
from _marching_cubes_lewiner import pseudosdf_mc_lewiner
import time

import torch
import numpy as np
import trimesh

POWERS_OF_2 = utils.POWERS_OF_2


def df_and_grad_to_input_cells(udf, grads, max_avg_distance=None, max_max_distance=None):
    """ Builds inputs cells for the network from the UDF and gradients. Each cell has 8 corners."""

    n_grid_samples = udf.shape[0]

    length = (n_grid_samples - 1) * (n_grid_samples - 1) * (n_grid_samples - 1)
    shape = (n_grid_samples - 1, n_grid_samples - 1, n_grid_samples - 1)

    udf_cells = torch.zeros((8, length))
    grad_cells = torch.zeros((8, length, 3))
    if (max_avg_distance is not None and max_max_distance is not None):
        z_indices, y_indices, x_indices = np.unravel_index(np.arange(length), shape)
        
        udf_cells[0] = udf[z_indices, y_indices, x_indices]
        udf_cells[1] = udf[z_indices, y_indices, x_indices + 1]
        udf_cells[2] = udf[z_indices, y_indices + 1, x_indices + 1]
        udf_cells[3] = udf[z_indices, y_indices + 1, x_indices]
        udf_cells[4] = udf[z_indices + 1, y_indices, x_indices]
        udf_cells[5] = udf[z_indices + 1, y_indices, x_indices + 1]
        udf_cells[6] = udf[z_indices + 1, y_indices + 1, x_indices + 1]
        udf_cells[7] = udf[z_indices + 1, y_indices + 1, x_indices]

        # Do the same for gradients
        grad_cells[0] = grads[z_indices, y_indices, x_indices]
        grad_cells[1] = grads[z_indices, y_indices, x_indices + 1]
        grad_cells[2] = grads[z_indices, y_indices + 1, x_indices + 1]
        grad_cells[3] = grads[z_indices, y_indices + 1, x_indices]
        grad_cells[4] = grads[z_indices + 1, y_indices, x_indices]
        grad_cells[5] = grads[z_indices + 1, y_indices, x_indices + 1]
        grad_cells[6] = grads[z_indices + 1, y_indices + 1, x_indices + 1]
        grad_cells[7] = grads[z_indices + 1, y_indices + 1, x_indices]

        within_thresholds = ((max_avg_distance is not None and max_max_distance is not None) and
                            (udf_cells.mean(dim=0) <= max_avg_distance) &
                            (udf_cells.max(dim=0).values <= max_max_distance))
        
        # Extract indices at which within_thresholds is True
        indices = torch.nonzero(within_thresholds).squeeze(1).int()

    else:
        indices = torch.arange(0, length, 1)

    return udf_cells, grad_cells, indices
        



def compute_pseudo_sdf(model, udf_and_grad_f, n_grid_samples=128):
    """Computes the pseudo-sdf of a mesh using a neural network model and a UDF and gradient function.
    
    Args:
        model: The neural network model.
        udf_and_grad_f: A function that takes query points and returns the UDF and gradients.
        n_grid_samples: The number of grid samples in each dimension.
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

        pseudo_sdf = torch.zeros(shape + (8,))

        # Build the input by putting the cells together
        input = torch.zeros((len(indices), 32))
        input[:,:8] = udf_cells[:,indices].T
        input[:,8:] = grad_cells[:,indices,:].permute(1,0,2).reshape(-1,24)


        udf2 = input[:,:8].clone()
        udf2 = udf2.to(device)

        # Normalize the UDF values with the voxel size. This is key for the network to work at multiple resolutions.
        input[:,:8] = input[:,:8] / voxel_size

        input = input.to(device)


        output = model(input)

        # Compute the pseudo-signs based on the network's output configuration
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