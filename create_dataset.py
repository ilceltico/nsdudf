import torch
import data
from torch.utils.data import DataLoader
import time
import os
import tqdm

# Creates a train/val dataset using a list of specified resolutions, so that less computation is needed during training or evaluation. It assumes that the objects are watertight, so it computes both UDF and ground truth SDF.

# A list of resolutions to create the dataset for
grid_points_list = [128]

# The location of a txt file containing the list of objects to use. The first 80% will be used for training and the rest for validation
data_list = "datasets/abc_obj_list_train_80.txt"
# The location of the meshes
dataset_location = "datasets/abc_0000_obj_v00"

# Base location to save the dataset
save_base_location = "datasets/ABC"

num_workers = 0

def main():
        
    for grid_points in grid_points_list:

        print(f"Creating dataset for {grid_points} grid points")

        voxel_size = 2.0 / (grid_points - 1)

        save_location = os.path.join(save_base_location, f"{grid_points}")

        # # Create directory if it does not exist
        os.makedirs(save_location, exist_ok=True)
        os.makedirs(os.path.join(save_location, "train"), exist_ok=True)    
        os.makedirs(os.path.join(save_location, "val"), exist_ok=True)


        print("Creating training set")
        training_dataset = data.GridDataset(dataset_location, data_list, grid_points, max_avg_distance=1.05*voxel_size, max_max_distance=1.74*voxel_size, class_balanced_weights=True, split="train", get_sdf=True)
        dataloader = DataLoader(training_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
        start = time.time()

        # Iterate using TQDM to show progress
        for inputs, gt, indices, class_weights, i in tqdm.tqdm(dataloader):
            # Check if it already exists
            if os.path.exists(os.path.join(os.path.join(save_location, "train"),f"{i[0]}.pt")):
                print(f"{i[0]+1}/{len(dataloader)} already exists")
                continue

            # Save the data (inputs, gt, indices, class weights, shape) to file using torch
            torch.save((inputs[0], gt[0], indices[0], class_weights[0], (grid_points-1,grid_points-1,grid_points-1)), os.path.join(os.path.join(save_location, "train"),f"{i[0]}.pt"))
     
        print(f"Time taken: {time.time() - start} seconds")



        print("Creating validation set")
        validation_dataset = data.GridDataset(dataset_location, data_list, grid_points, max_avg_distance=1.2*voxel_size, max_max_distance=2.0*voxel_size, class_balanced_weights=True, split="val", get_sdf=True)
        dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
        start = time.time()

        for inputs, gt, indices, class_weights, i in tqdm.tqdm(dataloader):
            # Check if it already exists
            if os.path.exists(os.path.join(os.path.join(save_location, "val"),f"{i[0]}.pt")):
                print(f"{i[0]+1}/{len(dataloader)} already exists")
                continue

            # Save the data (inputs, gt, indices, class weights, shape) to file using torch
            torch.save((inputs[0], gt[0], indices[0], class_weights[0], (grid_points-1,grid_points-1,grid_points-1)), os.path.join(os.path.join(save_location, "val"),f"{i[0]}.pt"))
        
        print(f"Time taken: {time.time() - start} seconds")


if __name__ == '__main__':
    main()
