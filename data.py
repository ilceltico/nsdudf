import os
import torch
import numpy as np
import trimesh
from trimesh.transformations import scale_matrix
import utils
import igl


POWERS_OF_2 = utils.POWERS_OF_2

class GridDataset(torch.utils.data.Dataset):
    def __init__(self, base_dir, split_file, grid_points, split = "train",  get_sdf = True, max_avg_distance = None, max_max_distance = None, class_balanced_weights = False):
        self.base_dir = base_dir
        self.split_file = split_file
        self.grid_points = grid_points
        self.get_sdf = get_sdf

        self.max_avg_distance = max_avg_distance
        self.max_max_distance = max_max_distance
        self.class_balanced_weights = class_balanced_weights

        lines = []
        with open(self.split_file, 'r') as f:
            lines = f.readlines()
        
        if split == "train":
            lines = lines[:int(len(lines)*0.8)]
        elif split == "val":
            lines = lines[int(len(lines)*0.8):]
        elif split == "all":
            lines = lines
        else:
            raise Exception("Invalid split")

        #Generate the file list. We assume there is only one obj file per subfolder
        self.file_list = []
        for line in lines:
            dir = os.path.join(self.base_dir, line.strip())
            #If it's a ply or obj file, add it to the list
            if dir.endswith(".obj") or dir.endswith(".ply"):
                self.file_list.append(dir)
            else:
                files = os.listdir(dir)
                files = [file for file in files if not file.startswith(".") and (file.endswith(".obj") or file.endswith(".ply"))]
                if len(files) == 0:
                    return Exception(f"No obj files found in directory {dir}")
                if len(files) > 1:
                    return Exception(f"Multiple obj files found in directory {dir}")
            
                self.file_list.append(os.path.join(dir, files[0]))
        
    def __len__(self):
        return len(self.file_list)
    
    # @profile
    def __getitem__(self, idx):
        mesh = trimesh.load_mesh(self.file_list[idx])

        query_points = np.zeros((self.grid_points ** 3,3))
        query_range = np.linspace(-1,1,self.grid_points)
        x_coords, y_coords, z_coords = np.meshgrid(query_range,query_range,query_range, indexing="ij") #"ij" indexing is important

        query_points[:,0] = x_coords.reshape(-1)
        query_points[:,1] = y_coords.reshape(-1)
        query_points[:,2] = z_coords.reshape(-1)

        if isinstance(mesh, trimesh.Scene):
            if len(mesh.geometry) == 0:
                mesh = None  # empty scene
            else:
                # we lose texture information here
                mesh = trimesh.util.concatenate(
                    tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                        for g in mesh.geometry.values()))
                
        mesh.apply_translation(-mesh.bounds.mean(axis=0))

        scale = scale_matrix(1.99999 / max(mesh.extents)) #Keeping 2.0 causes some faces to be exactly on the edge of the box, which is potentially problematic
        mesh.apply_transform(scale)

        if self.get_sdf:
            sdf, facet_indices, closest_points = igl.signed_distance(query_points, mesh.vertices, mesh.faces)
        else:
            # This is actually an unisgned distance
            sdf, facet_indices, closest_points = igl.point_mesh_squared_distance(query_points, mesh.vertices, mesh.faces)
            sdf = np.sqrt(sdf)

        sdf = torch.Tensor(sdf)

        # udf = torch.abs(sdf)
        udf_grads = query_points - closest_points
        udf_grads = torch.Tensor(udf_grads)
        udf_grads_normalized = udf_grads / torch.linalg.norm(udf_grads, axis=1).reshape(-1,1)

        sdf = sdf.reshape(self.grid_points, self.grid_points, self.grid_points)

        udf_grads_normalized = udf_grads_normalized.reshape(self.grid_points, self.grid_points, self.grid_points, -1)

        # Some query points are exactly on the surface and produce NaN gradients
        # The correct SDF gradient would be the surface normal on that point, but here we assume to use UDF gradients for now, so we set them to zero.
        udf_grads_normalized = torch.nan_to_num(udf_grads_normalized, nan=0.0)

        # Precheck the values to filter out unwanted ones and create an index list of desired ones
        # Also computing the class balancing weights here
        # Slow TODO
        class_weights = torch.ones((128))
        length = (self.grid_points - 1) * (self.grid_points - 1) * (self.grid_points - 1)
        shape = (self.grid_points - 1, self.grid_points - 1, self.grid_points - 1)

        sdf_values = torch.zeros((8, length))
        udf_values = torch.zeros((8, length))
        grad_values = torch.zeros((8, length, 3))
        if ((self.max_avg_distance is not None and self.max_max_distance is not None) or self.class_balanced_weights):
            z_indices, y_indices, x_indices = np.unravel_index(np.arange(length), shape)
            
            sdf_values[0] = sdf[z_indices, y_indices, x_indices]
            sdf_values[1] = sdf[z_indices, y_indices, x_indices + 1]
            sdf_values[2] = sdf[z_indices, y_indices + 1, x_indices + 1]
            sdf_values[3] = sdf[z_indices, y_indices + 1, x_indices]
            sdf_values[4] = sdf[z_indices + 1, y_indices, x_indices]
            sdf_values[5] = sdf[z_indices + 1, y_indices, x_indices + 1]
            sdf_values[6] = sdf[z_indices + 1, y_indices + 1, x_indices + 1]
            sdf_values[7] = sdf[z_indices + 1, y_indices + 1, x_indices]

            grad_values[0] = udf_grads_normalized[z_indices, y_indices, x_indices]
            grad_values[1] = udf_grads_normalized[z_indices, y_indices, x_indices + 1]
            grad_values[2] = udf_grads_normalized[z_indices, y_indices + 1, x_indices + 1]
            grad_values[3] = udf_grads_normalized[z_indices, y_indices + 1, x_indices]
            grad_values[4] = udf_grads_normalized[z_indices + 1, y_indices, x_indices]
            grad_values[5] = udf_grads_normalized[z_indices + 1, y_indices, x_indices + 1]
            grad_values[6] = udf_grads_normalized[z_indices + 1, y_indices + 1, x_indices + 1]
            grad_values[7] = udf_grads_normalized[z_indices + 1, y_indices + 1, x_indices]

            udf_values = torch.abs(sdf_values)
            within_thresholds = ((self.max_avg_distance is not None and self.max_max_distance is not None) and
                                (udf_values.mean(dim=0) <= self.max_avg_distance) &
                                (udf_values.max(dim=0).values <= self.max_max_distance))
            
            # Extract indices at which within_thresholds is True
            indices = torch.nonzero(within_thresholds).squeeze(1).int()

            reference_sdf_values = sdf_values[:, within_thresholds] * torch.sign(sdf_values[0, within_thresholds])
            reference_sdf_values[reference_sdf_values < 0] = 0
            reference_sdf_values[reference_sdf_values > 0] = 1
            gt_one_hot_numbers = reference_sdf_values[1:,:].transpose(1,0).matmul(POWERS_OF_2.float()).int()

            if self.class_balanced_weights:
                #Count the different classes in gt_one_hot_numbers
                class_weights = torch.bincount(gt_one_hot_numbers, minlength=128)
                class_weights = 1. / class_weights
                class_weights[torch.isinf(class_weights)] = 0
        else:
            indices = torch.arange(0, length, 1)

        # Creates the input tensor for the network
        inputs = torch.hstack((udf_values.permute(1,0), grad_values.permute(1,0,2).reshape(length,-1)))

        # Compute the ground truth sign configurations
        # Assume that the first corner is the reference corner and is positive
        base_sign = torch.sign(sdf_values[0,:])
        # I consider 0 to be positive
        base_sign[base_sign == 0] = 1
        # If the reference corner is negative, flip the sign of all corners
        reference_sdf_values = sdf_values[:8,:] * base_sign
        reference_sdf_values[reference_sdf_values >= 0] = 1
        reference_sdf_values[reference_sdf_values < 0] = 0
        # Compute the one-hot configuration of the ground truth. The reference corner is not considered because it is assumed positive.
        gt_one_hot_numbers = reference_sdf_values[1:,:] * POWERS_OF_2.view(-1,1)
        gt_one_hot_numbers = gt_one_hot_numbers.sum(axis=0).int() # A list of numbers between 0 and 127, one for each cell
        # Now we produce the one-hot encoding of the ground truth
        gt = torch.zeros((2**7, length))
        gt[gt_one_hot_numbers, torch.arange(length)] = 1

        # Filter the inputs and ground truth based on the indices
        inputs = inputs[indices]
        gt = gt[:, indices]
        gt = gt.permute(1,0)
        
        return inputs, gt, indices, class_weights, idx
    

class PrecomputedGridDataset(torch.utils.data.Dataset):
    def __init__(self, base_dirs, noise_udf = 0.0, noise_udf_type = "add", noise_grad = 0.0, noise_grad_type = "add", noise_grad_swap = None):
        self.base_dirs = base_dirs

        # List the toch files in the directory
        self.file_list = []
        for base_dir in self.base_dirs:
            base_dir_list = os.listdir(base_dir)
            #Sort it numerically
            base_dir_list = sorted(base_dir_list, key=lambda x: int(x.split(".")[0]))
            for file in base_dir_list:
                if file.endswith(".pt"):
                    self.file_list.append(os.path.join(base_dir, file))

        self.noise_udf = noise_udf
        self.noise_udf_type = noise_udf_type
        self.noise_grad = noise_grad
        self.noise_grad_type = noise_grad_type
        self.noise_grad_swap = noise_grad_swap
        
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):

        input, gt, indices, class_weights, shape = torch.load(self.file_list[idx])

        gradient_start = 8

        # Gaussian noise on the UDF
        if (self.noise_udf != 0.0):
            if self.noise_udf_type == "add":
                noise = torch.randn(input[:,:gradient_start].shape)*self.noise_udf
                input[:,:gradient_start] = input[:,:gradient_start] + noise
            elif self.noise_udf_type == "scale":
                noise = torch.randn(input[:,:gradient_start].shape)*self.noise_udf
                input[:,:gradient_start] = input[:,:gradient_start] * (1+ noise)
            elif self.noise_udf_type == "add_exp":
                scale = torch.exp(-(input[:,:gradient_start].detach() / self.noise_udf)**2) * self.noise_udf
                noise = torch.randn(input[:,:gradient_start].shape) * scale
                input[:,:gradient_start] = input[:,:gradient_start] + noise.abs()
            elif self.noise_udf_type == "scale_exp":
                scale = torch.exp(-(input[:,:gradient_start].detach() / self.noise_udf)**2)
                noise = torch.randn(input[:,:gradient_start].shape) * scale
                input[:,:gradient_start] = input[:,:gradient_start] * (1+ noise).abs()
            else:
                raise Exception("Unknown noise type")
        
        # Gaussian noise on the gradients
        if (self.noise_grad != 0.0):
            noise = torch.randn(input[:,gradient_start:].shape)*self.noise_grad
            if self.noise_grad_type == "add":
                input[:,gradient_start:] = input[:,gradient_start:] + noise
            elif self.noise_grad_type == "scale":
                input[:,gradient_start:] = input[:,gradient_start:] * (1+ noise)
            elif self.noise_grad_type == "add_exp":
                scale = torch.exp(-(input[:,gradient_start:].detach() / self.noise_grad)**2) * self.noise_grad
                noise = torch.randn(input[:,gradient_start:].shape) * scale
                input[:,gradient_start:] = input[:,gradient_start:] + noise
            elif self.noise_grad_type == "scale_exp":
                scale = torch.exp(-(input[:,gradient_start:].detach() / self.noise_grad)**2)
                noise = torch.randn(input[:,gradient_start:].shape) * scale
                input[:,gradient_start:] = input[:,gradient_start:] * (1+ noise)
            else:
                raise Exception("Unknown noise type")
            
        # Gradient swapping (randomly swap gradient direction, higher chance for low UDF values)
        grad_swap = self.noise_grad_swap
        if grad_swap is not None and grad_swap > 0.:
            prob = 0.5 * torch.exp(-(input[:,:gradient_start].detach() / grad_swap)**2)
            swap = torch.rand(input[:,:gradient_start].shape) < prob
            swap = 1 - 2 * swap  # from proba to sign
            swap = swap.repeat_interleave(3).reshape(input[:,gradient_start:].shape)
            input[:,gradient_start:] = input[:,gradient_start:] * swap
                
        return input, gt, indices, class_weights, shape, idx
