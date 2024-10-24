import torch
import core.data as data
from torch.utils.data import DataLoader
import core.models as models
import time
import numpy as np
import trimesh
import os
import logging
import argparse
from tqdm import tqdm

import sys
sys.path.append('custom_mc')
from _marching_cubes_lewiner import pseudosdf_mc_lewiner


# When training on CPU, we suggest to set it to 0
num_workers = 8

POWERS_OF_2 = data.POWERS_OF_2

def main():

    #Parse arguments from command line
    parser = argparse.ArgumentParser(description='Train a neural network to predict the sign configuration of a UDF voxel grid')
    parser.add_argument('--save_base_location', type=str, default="./experiments", help='Base save location for the experiment')
    parser.add_argument('--dataset_location', type=str, default="./datasets/ABC", help='Location of the training dataset')
    parser.add_argument('--lr', type=float, default=5e-3, help='Learning rate')
    parser.add_argument('--grid_points_list', type=int, nargs='+', default=[128], help='List of resolutions to train on. Default is 128 only.')
    parser.add_argument('--noise_udf', type=float, default=1.0, help='Gaussian noise to augment the distance field. Set to zero to disable.')
    parser.add_argument('--noise_udf_type', type=str, default="scale", help='add | scale | add_exp | scale_exp')
    parser.add_argument('--noise_grad', type=float, default=1.0, help='Gaussian noise to augment the gradients. Set to zero to disable.')
    parser.add_argument('--noise_grad_type', type=str, default="scale", help='add | scale | add_exp | scale_exp')
    parser.add_argument('--noise_grad_swap', type=float, default=0.0, help='Swap the gradients with a probability. Set to zero to disable.')
    parser.add_argument('--renormalize_gradients', default=False, action="store_true", help='Normalizes noisy training gradients to unitary norm')
    parser.add_argument('--balanced', default=False, action="store_true", help='Rebalance the CE loss using the class weights')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')

    parser.add_argument('--device', type=str, default="cpu", help='Device to use (cpu, mps, cuda)')
    parser.add_argument('--batch_size', type=int, default=10000, help='Batch size. Has no effect on results, only on memory usage.')
    parser.add_argument('--export_meshes', action="store_true", help='Export meshes during evaluation')
    parser.add_argument('--no-export_meshes', dest='export_meshes', action='store_false', help='Do not export meshes during evaluation')
    parser.set_defaults(export_meshes=True)

    # The following arguments require a change in the corresponding parameters of the extraction algorithm if they are modified.
    parser.add_argument('--not_normalized', action="store_true", help='Do not normalize the UDF values based on resolution, becoming less robust to resolution changes.')
    parser.add_argument('--out7', action="store_true", help='Use 7 outputs instead of the default 128.')
    parser.add_argument('--nograd', action="store_true", help='Do not use gradients in the input, losing valuable information.')
    
    args = parser.parse_args()

    device = torch.device(args.device)
    normalize_df = not args.not_normalized
    global POWERS_OF_2
    POWERS_OF_2 = POWERS_OF_2.to(device)

    today = time.strftime("%Y%m%d-%H%M%S")
    grid_points_list_string = "_".join([str(x) for x in args.grid_points_list])
    save_location = os.path.join(args.save_base_location, f"{today}_{grid_points_list_string}{'_normalized' if normalize_df else ''}{f'_noiseudf{args.noise_udf_type}{args.noise_udf}' if args.noise_udf > 0 else ''}{f'_noisegrad{args.noise_grad_type}{args.noise_grad}' if args.noise_grad > 0 else ''}{'renormgrads' if args.renormalize_gradients else ''}{'_balanced' if args.balanced else ''}{'_7out' if args.out7 else ''}{f'_{args.epochs}ep'}{'_nograd' if args.nograd else ''}")

    os.makedirs(save_location, exist_ok=True)

    # Setup the logger with the save directory
    logger = logging.getLogger()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(save_location, "log.txt")),
            logging.StreamHandler()
        ]
    )

    logging.info(f"Arguments: {args}")

    # Copy the training code to the save location for reproducibility
    os.makedirs(os.path.join(save_location, "code"), exist_ok=True)
    os.system(f"cp train.py {os.path.join(save_location, 'code')}")
    # Copy also the core
    os.system(f"cp -r core {os.path.join(save_location, 'code')}")

    if args.nograd:
        input_dims = 8
    else:
        input_dims = 32

    if args.out7:
        model = models.MLP([input_dims,1024,1024,7], torch.nn.LeakyReLU)
    else:
        model = models.MLP([input_dims,1024,1024,128], torch.nn.LeakyReLU)
    model = model.to(device)

    optimizer = torch.optim.Adam(
            [
                {
                    "params": model.parameters(),
                    "lr": args.lr,
                }
            ]
        )

    train(logger, model, optimizer, args.epochs, args.grid_points_list, args.dataset_location, args.batch_size, args.noise_udf, args.noise_udf_type, args.noise_grad, args.noise_grad_type, args.noise_grad_swap, normalize_df, args.renormalize_gradients, args.balanced, args.out7, args.nograd)
    
    evaluate(logger, model, args.batch_size, [os.path.join(args.dataset_location, "128/train")], 128, normalize_df, args.export_meshes, os.path.join(save_location,"exports/train/"), "train128", args.renormalize_gradients, args.out7, args.nograd)
    evaluate(logger, model, args.batch_size, [os.path.join(args.dataset_location, "128/val")], 128, normalize_df, args.export_meshes, os.path.join(save_location,"exports/val/"), "val128", args.renormalize_gradients, args.out7, args.nograd)
    
    print(f"Done")


def train(logger, model, optimizer, epochs, grid_points_list, dataset_location_base, batch_size, noise_udf, noise_udf_type, noise_grad, noise_grad_type, noise_grad_swap, normalize_df, renormalize_gradients, balanced, out7, nograd):
    device = next(model.parameters()).device

    logger.log(logging.INFO, f"Training model on {device} for {epochs} epochs at resolutions {grid_points_list}")

    training_start = time.time()

    dataloaders = []
    for grid_points in grid_points_list:
        dataset_location = os.path.join(dataset_location_base, f"{grid_points}")
        training_dataset = data.PrecomputedGridDataset([os.path.join(dataset_location, "train")], noise_udf, noise_udf_type, noise_grad, noise_grad_type, noise_grad_swap)
        dataloader = DataLoader(training_dataset, batch_size=1, shuffle=True, num_workers=num_workers)
        dataloaders.append(dataloader)


    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")

        epoch_loss = 0.0
        epoch_num_valid = 0
        epoch_total = 0
        epoch_non_empty_num_valid = 0
        epoch_non_empty_total = 0
        epoch_close_num_valid = 0
        epoch_close_total = 0

        epoch_start = time.time()

        for dataloader, grid_points in zip(dataloaders, grid_points_list):
            if len(dataloaders) > 1:
                print(f"Training on dataloader {grid_points}")

            dataloader_loss = 0.0
            dataloader_num_valid = 0
            dataloader_total = 0
            dataloader_non_empty_num_valid = 0
            dataloader_non_empty_total = 0
            dataloader_close_num_valid = 0
            dataloader_close_total = 0

            voxel_size = 2.0 / (grid_points - 1)

            for input_dataset, gt_dataset, indices, class_weights, sdf_shape, i in tqdm(dataloader):
                model.train()
                optimizer.zero_grad()

                input_dataset = input_dataset.to(device)
                gt_dataset = gt_dataset.to(device)

                shape_loss = 0.0    
                shape_num_valid = 0
                shape_total = 0
                shape_non_empty_num_valid = 0
                shape_non_empty_total = 0
                shape_close_num_valid = 0
                shape_close_total = 0

                class_weights = class_weights.to(device)
                if balanced:
                    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights[0])
                else:
                    loss_fn = torch.nn.CrossEntropyLoss()

                if out7:
                    loss_fn = torch.nn.BCEWithLogitsLoss()


                # You can move batching here if needed. We did not need it for the experiments so we only use it below.
                for batch_i in range(1):
                    input = input_dataset[0]
                    gt = gt_dataset[0]

                    if nograd:
                        input = input[:,:8]

                    if normalize_df:
                        input[:,:8] = input[:,:8] / voxel_size

                    # Normalize gradients to unitary norm, which is the assumption of this method
                    if renormalize_gradients:
                        input[:,8:] = (input[:,8:].reshape(-1,3) / torch.linalg.norm(input[:,8:].reshape(-1,3), axis=1).reshape(-1,1)).reshape(-1,input.shape[1]-8)

                    # Batching, only used for memory purposes, does not affect the training
                    output = torch.zeros((input.shape[0],7 if out7 else 128)).to(device)
                    batch_head = 0
                    while batch_head < input.shape[0]:
                        batch_tail = min(batch_head + batch_size, input.shape[0])
                        output[batch_head:batch_tail] = model(input[batch_head:batch_tail])
                        batch_head += batch_size
                    
                    if not out7:
                        batch_loss = loss_fn(output, gt)
                        pred_index = torch.argmax(output, axis=1)
                        gt_index = torch.argmax(gt, axis=1)
                        
                    else:
                        gt_sign_config = gt.argmax(axis=1).unsqueeze(-1).bitwise_and(POWERS_OF_2).ne(0).int().float()

                        batch_loss = loss_fn(output, gt_sign_config)

                        pred_index = ((torch.sigmoid(output) > 0.5).int() * POWERS_OF_2).sum(axis=1)
                        gt_index = torch.argmax(gt, axis=1)

                        
                    num_valid = (pred_index == gt_index).sum()
                    total = pred_index.shape[0]

                    non_empty_num_valid = ((pred_index == gt_index)*(gt_index != 127)).sum()
                    non_empty_total = (gt_index != 127).sum()

                    udf = input[:,:8]
                    #Same closeness conditions as MeshUDF
                    if normalize_df:
                        close_num_valid = ((pred_index == gt_index) * (udf.mean(axis=1) < 1.05) * (udf.max(axis=1).values <= 1.74)).sum()
                        close_total = ((udf.mean(axis=1) < 1.05) * (udf.max(axis=1).values <= 1.74)).sum()
                    else:
                        close_num_valid = ((pred_index == gt_index) * (udf.mean(axis=1) < 1.05 * voxel_size) * (udf.max(axis=1).values <= 1.74 * voxel_size)).sum()
                        close_total = ((udf.mean(axis=1) < 1.05 * voxel_size) * (udf.max(axis=1).values <= 1.74 * voxel_size)).sum()


                    batch_loss.backward()
                    shape_loss += batch_loss
                    shape_num_valid += num_valid
                    shape_total += total
                    shape_non_empty_num_valid += non_empty_num_valid
                    shape_non_empty_total += non_empty_total
                    shape_close_num_valid += close_num_valid
                    shape_close_total += close_total


                dataloader_loss += shape_loss
                dataloader_num_valid += shape_num_valid
                dataloader_total += shape_total
                dataloader_non_empty_num_valid += shape_non_empty_num_valid
                dataloader_non_empty_total += shape_non_empty_total
                dataloader_close_num_valid += shape_close_num_valid
                dataloader_close_total += shape_close_total

                optimizer.step()
            
            if len(dataloaders) > 1:
                logger.log(logging.INFO, f"Dataloader {grid_points}, loss = {dataloader_loss}, close correct {dataloader_close_num_valid}/{dataloader_close_total} {dataloader_close_num_valid/dataloader_close_total*100}%, correct {dataloader_num_valid}/{dataloader_total} {dataloader_num_valid/dataloader_total * 100}%, non empty correct {dataloader_non_empty_num_valid}/{dataloader_non_empty_total} {dataloader_non_empty_num_valid/dataloader_non_empty_total * 100}%, time {time.time() - epoch_start}")

            epoch_loss += dataloader_loss
            epoch_num_valid += dataloader_num_valid
            epoch_total += dataloader_total
            epoch_non_empty_num_valid += dataloader_non_empty_num_valid
            epoch_non_empty_total += dataloader_non_empty_total
            epoch_close_num_valid += dataloader_close_num_valid
            epoch_close_total += dataloader_close_total

        logger.log(logging.INFO, f"End of epoch {epoch}, loss = {epoch_loss}, close correct {epoch_close_num_valid}/{epoch_close_total} {epoch_close_num_valid/epoch_close_total*100}%, correct {epoch_num_valid}/{epoch_total} {epoch_num_valid/epoch_total * 100}%, non empty correct {epoch_non_empty_num_valid}/{epoch_non_empty_total} {epoch_non_empty_num_valid/epoch_non_empty_total * 100}%, time {time.time() - epoch_start}")
    
    logger.log(logging.INFO, f"Training done in {time.time() - training_start} seconds")

def evaluate(logger, model, batch_size, data_paths, num_grid_points, normalize_df, export_meshes, save_location, evaluation_name, renormalize_gradients, out7, nograd):
    device = next(model.parameters()).device
    voxel_size = 2.0 / (num_grid_points - 1)

    logger.log(logging.INFO, f"Evaluating model on {device}, evaluation name: {evaluation_name}")

    #Validation
    model.eval()

    # Unweighted CE loss
    loss_fn = torch.nn.CrossEntropyLoss()

    if model.layers[-2].weight.shape[0] == 7:
        loss_fn = torch.nn.BCEWithLogitsLoss()

    if export_meshes:
        #Create save location
        os.makedirs(os.path.join(save_location, evaluation_name), exist_ok=True)
    
    with torch.no_grad():

        evaluation_dataset = data.PrecomputedGridDataset(data_paths)
        evaluation_dataloader = DataLoader(evaluation_dataset, batch_size=1, shuffle=False, num_workers=num_workers)

        for epoch in range(1):

            epoch_loss = 0.0
            epoch_num_valid = 0
            epoch_total = 0
            epoch_non_empty_num_valid = 0
            epoch_non_empty_total = 0
            epoch_close_num_valid = 0
            epoch_close_total = 0

            start = time.time()

            for input_dataset, gt_dataset, indices, class_weights, sdf_shape, i in tqdm(evaluation_dataloader):
                input_dataset = input_dataset.to(device)
                gt_dataset = gt_dataset.to(device)

                shape_loss = 0.0    
                shape_num_valid = 0
                shape_total = 0
                shape_non_empty_num_valid = 0
                shape_non_empty_total = 0
                shape_close_num_valid = 0
                shape_close_total = 0

                if export_meshes:
                    pseudo_sdf = torch.zeros(tuple(sdf_shape) + (8,))

                # You can move batching here if needed. We did not need it for the experiments so we only use it below.
                for shape_i in range(1):
                    input = input_dataset[0]
                    gt = gt_dataset[0]

                    if nograd:
                        input = input[:,:8]

                    udf = input[:,:8].clone()

                    if normalize_df:
                        input[:,:8] = input[:,:8] / voxel_size

                    output = torch.zeros((input.shape[0],7 if out7 else 128)).to(device)
                    batch_head = 0
                    while batch_head < input.shape[0]:
                        batch_tail = min(batch_head + batch_size, input.shape[0])
                        output[batch_head:batch_tail] = model(input[batch_head:batch_tail])
                        batch_head += batch_size

                    if not out7:
                        batch_loss = loss_fn(output, gt)
                        pred_index = torch.argmax(output, axis=1)
                        gt_index = torch.argmax(gt, axis=1)
                        
                    else:
                        gt_sign_config = gt.argmax(axis=1).unsqueeze(-1).bitwise_and(POWERS_OF_2).ne(0).int().float()

                        batch_loss = loss_fn(output, gt_sign_config)

                        pred_index = ((torch.sigmoid(output) > 0.5).int() * POWERS_OF_2).sum(axis=1)
                        gt_index = torch.argmax(gt, axis=1)

                    num_valid = (pred_index == gt_index).sum()
                    total = pred_index.shape[0]

                    non_empty_num_valid = ((pred_index == gt_index)*(gt_index != 127)).sum()
                    non_empty_total = (gt_index != 127).sum()

                    #Same closeness conditions as MeshUDF
                    close_num_valid = ((pred_index == gt_index) * (udf.mean(axis=1) < 1.05 * voxel_size) * (udf.max(axis=1).values <= 1.74 * voxel_size)).sum()
                    close_total = ((udf.mean(axis=1) < 1.05 * voxel_size) * (udf.max(axis=1).values <= 1.74 * voxel_size)).sum()

                    if export_meshes:
                        if output.shape[1] == 128:
                            pseudo_signs = output.argmax(axis=1).unsqueeze(-1).bitwise_and(POWERS_OF_2).ne(0).int()
                        else:
                            pseudo_signs = (torch.sigmoid(output) > 0.5).int()
                        pseudo_signs[pseudo_signs == 0] = -1

                        pseudo_sdf[np.unravel_index(indices[0], sdf_shape)] = torch.hstack((udf[:,0].reshape(-1,1), udf[:,1:8] * pseudo_signs)).cpu()

                    shape_loss += batch_loss
                    shape_num_valid += num_valid
                    shape_total += total
                    shape_non_empty_num_valid += non_empty_num_valid
                    shape_non_empty_total += non_empty_total
                    shape_close_num_valid += close_num_valid
                    shape_close_total += close_total

                epoch_loss += shape_loss
                epoch_num_valid += shape_num_valid
                epoch_total += shape_total
                epoch_non_empty_num_valid += shape_non_empty_num_valid
                epoch_non_empty_total += shape_non_empty_total
                epoch_close_num_valid += shape_close_num_valid
                epoch_close_total += shape_close_total

                #Mesh the resulting pseudo-sdf using Marching Cubes
                if export_meshes:
                    try:
                        vertices, faces, normals, values = pseudosdf_mc_lewiner(pseudo_sdf.cpu().detach().numpy(), spacing=[voxel_size] * 3)
                    except:
                        print(f"Failed to mesh {i.numpy()[0]}")
                        continue

                    vertices = vertices - 1 # Since voxel origin is [-1,-1,-1]
                    reconstructed_mesh = trimesh.Trimesh(vertices, faces)

                    reconstructed_mesh.export(os.path.join(os.path.join(save_location, evaluation_name), f"{evaluation_name}_{num_grid_points}_{i.numpy()[0]}.ply"))

                
            logger.log(logging.INFO, f"{evaluation_name} loss = {epoch_loss}, close correct {epoch_close_num_valid}/{epoch_close_total} {epoch_close_num_valid/epoch_close_total*100}%, correct {epoch_num_valid}/{epoch_total} {epoch_num_valid/epoch_total * 100}%, non empty correct {epoch_non_empty_num_valid}/{epoch_non_empty_total} {epoch_non_empty_num_valid/epoch_non_empty_total * 100}%, time {time.time() - start}")



if __name__ == '__main__':
    main()
