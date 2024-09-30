import torch
import models

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
    print(f"You are using {device}. Loading model from {model_path}")
    
    model = models.MLP([32,1024,1024,128], torch.nn.LeakyReLU)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model
