import os
import numpy as np
import tqdm
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from pathlib import Path
from time import time

import Utils.DataLoader as dloader
import Utils.Rendering as render
import Model.VanillaModel as NerfModel


def training(model, nerf_render, data_loader, optimizer, scheduler, tn, tf, num_epochs, device='cpu'):
    """
    A function for training the Vanilla NeRF model.
    :param model: a torch nn.Module object that defines the model object to be used for training
    :param nerf_render: an object that contains methods for rendering the NeRF
    :param data_loader: a torch dataloader object that will be used for training the model
    :param optimizer: a torch object that defines the optimizer for training the model
    :param scheduler: a torch object that defines a scheduler
    :param tn: the lower bound of the point sampling interval (see eq (1) in NeRF paper)
    :param tf: the upper bound of the point sampling interval (see eq (1) in NeRF paper)
    :param num_epochs: an int parameter defining the amount of epochs for training the model
    :param device: a string parameter that defines the device to be used for training: 'cpu' or 'cuda'
    :return: numpy array with training losses and a trained torch model
    """
    t1 = time()

    training_loss = []
    for e in tqdm.trange(num_epochs):
        for batch in data_loader:
            o = batch[:, :3].to(device)
            d = batch[:, 3:6].to(device)
            target = batch[:, 6:].to(device)

            prediction = nerf_render.render_view(model, o, d, tn, tf)
            loss = ((prediction - target)**2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            training_loss.append(loss.detach().cpu().numpy())
        print("[INFO] Training loss: ", loss.detach().cpu().numpy())
        scheduler.step()

    t2 = time()
    print("It took: ", (t2-t1)/60, ' min')

    return training_loss, model


if __name__ == '__main__':
    root = Path(os.path.realpath(__file__)).parent

    data = r"D:\Myfiles\Projects\OnlineCourses\Udemy_nerf\dataset\fox"
    camera_intrinsic_train_data = data + "/train/intrinsics"
    camera_positional_train_data = data + "/train/pose"
    imgs_train_data = data + "/imgs/train"
    model_name = "nerf_model_v2"
    loss_file = "train_loss_nerf"

    device = "cuda"
    batch_size = 1024
    img_res_x = 400
    img_res_y = 400
    num_bins = 100
    num_epochs = 1
    chunk_size = 10
    learning_rate = 0.001
    tn = 8
    tf = 12
    train = True

    # loading camera parameters (transforms + intrinsics), images for training the model
    print("[INFO] Loading training data.")
    loader = dloader.DataSetLoader()
    train_imgs = loader.load_img_dataset(imgs_train_data)
    train_cam_intr = loader.load_cameras_intrinsics(camera_intrinsic_train_data)
    train_cam_poses = loader.load_cameras_positions(camera_positional_train_data)
    print("[INFO] images arr., shape:            ", train_imgs.shape)
    print("[INFO] cameras intr. arr., shape:     ", train_cam_intr.shape)
    print("[INFO] cameras transform arr., shape: ", train_cam_poses.shape)
    print("[INFO] Done.\n")

    # preparing data that will be used for training NeRF: arrays with cameras rays origins and direction vectors;
    # pixel arrays generated from input images
    print("[INFO] Generating camera rays: origins & directions arrays.")
    NerfRender = render.NerfRender(img_res_x, img_res_y, num_bins, device)
    train_rays_o, train_rays_d = NerfRender.generate_rays(train_cam_poses, train_cam_intr)
    train_target_px_vals = NerfRender.generate_target_pixel_arr(train_imgs)
    print("[INFO] Done.\n")

    model_dir = root / "torch_models"
    model_dir.mkdir(exist_ok=True)

    loss_dir = root / "loss"
    loss_dir.mkdir(exist_ok=True)

    if train:
        print("[INFO] Start training of the model.")
        # preparing training data loaders
        data_tensor = torch.cat((train_rays_o.reshape(-1, 3), train_rays_d.reshape(-1, 3), train_target_px_vals.reshape(-1, 3)), dim=1)
        dataloader = DataLoader(data_tensor, batch_size=batch_size, shuffle=True)

        # training warm up: training only the middle part of the dataset, synthetic data only
        data_tensor_warmup = torch.cat((train_rays_o.reshape(90, 400, 400, 3)[:, 100:300, 100:300, :].reshape(-1, 3),
                                        train_rays_d.reshape(90, 400, 400, 3)[:, 100:300, 100:300, :].reshape(-1, 3),
                                        train_target_px_vals.reshape(90, 400, 400, 3)[:, 100:300, 100:300, :].reshape(-1, 3)), dim=1)
        dataloader_warmup = DataLoader(data_tensor_warmup, batch_size=batch_size, shuffle=True)

        # Setting up the model and it's training
        model = NerfModel.VanillaNerfModel().to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10], gamma=0.5)

        _, model = training(model, NerfRender, dataloader_warmup, optimizer, scheduler, tn, tf, 1, device=device)
        training_loss, model = training(model, NerfRender, dataloader, optimizer, scheduler, tn, tf, num_epochs, device=device)

        print("[INFO] Done. \n")
        print("[INFO] Saving data.")

        torch.save(model.state_dict(), model_dir.as_posix() + "/" + model_name + ".pt")
        np.savetxt(loss_dir.as_posix() + "/" + loss_file + ".csv", training_loss, delimiter=',')
        print("[INFO] Done. \n")
    else:
        print("[INFO] Loading model: ", model_name + ".pt")
        model = NerfModel.VanillaNerfModel().to(device)
        model.load_state_dict(torch.load(model_dir.as_posix() + "/" + model_name + ".pt"))
        training_loss = np.loadtxt(loss_dir.as_posix() + "/" + loss_file + ".csv", delimiter=',')
        print("[INFO] Done. \n")

    # rendering the nerf model after training
    img = NerfRender.generate_view(model, train_rays_o[10], train_rays_d[10], tn, tf, chunk_size)
    plt.imshow(img)
    plt.show()
