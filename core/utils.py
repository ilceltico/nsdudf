import torch
import core.models as models
import numpy as np

POWERS_OF_2 = 2**torch.arange(7)

def get_query_points(bbox, N):
    """Creates a list of points on a regular grid."""
    coords = []
    for i in range(len(bbox[0])):
        coords.append(torch.linspace(bbox[0][i], bbox[1][i], N))
    coords = torch.meshgrid(*coords, indexing='ij')
    coords = torch.stack(coords, dim=-1)
    return coords

def load_model(model_path, device):
    print(f"You are using {device.upper()}. Loading model from {model_path}")
    
    model = models.MLP([32,1024,1024,128], torch.nn.LeakyReLU)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def normalize(v, dim=-1):
    norm = torch.linalg.norm(v, axis=dim, keepdims=True)
    norm[norm == 0] = 1
    return v / norm

def df_and_grad_to_input_cells(udf, grads, max_avg_distance=None, max_max_distance=None):
    """ Builds inputs cells for the network from the UDF and gradients, and filters out cells that are far from the surface. Each cell has 8 corners.
        Return a list of udf cell, a list of gradient cells, and a list of indices of the cells that are within the thresholds."""

    n_grid_samples = udf.shape[0]

    length = (n_grid_samples - 1) * (n_grid_samples - 1) * (n_grid_samples - 1)
    shape = (n_grid_samples - 1, n_grid_samples - 1, n_grid_samples - 1)

    udf_cells = torch.zeros((8, length))
    grad_cells = torch.zeros((8, length, 3))
    
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

    if (max_avg_distance is not None and max_max_distance is not None):
        within_thresholds = ((max_avg_distance is not None and max_max_distance is not None) and
                            (udf_cells.mean(dim=0) <= max_avg_distance) &
                            (udf_cells.max(dim=0).values <= max_max_distance))
    else:
        within_thresholds = torch.ones(length, dtype=torch.bool)
        
    # Extract indices at which within_thresholds is True
    indices = torch.nonzero(within_thresholds).squeeze(1).int()


    return udf_cells, grad_cells, indices
        