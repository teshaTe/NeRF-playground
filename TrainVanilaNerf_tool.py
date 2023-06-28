import tqdm
import torch

from torch.utils.data import DataLoader
from time import time

import Utils.DataLoader as dloader
import Utils.Rendering as render
import Model.VanillaModel as NerfModel


def training(model, data_loader, optimizer, scheduler, tn, tf, res_x, res_y, num_bins, num_epochs, device='cpu'):
    """

    :param model:
    :param data_loader:
    :param optimizer:
    :param scheduler:
    :param tn:
    :param tf:
    :param res_x:
    :param res_y:
    :param num_bins:
    :param num_epochs:
    :param device:
    :return:
    """
    t1 = time()

    NerfRender = render.NerfRender(res_x, res_y, num_bins, device)
    training_loss = []
    for e in tqdm.trange(num_epochs):
        for batch in data_loader:
            o = batch[:, :3].to(device)
            d = batch[:, 3:6].to(device)
            target = batch[:, 6:].to(device)

            prediction = NerfRender.render_view(model, o, d, tn, tf)
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
    data = r"D:\Myfiles\Projects\OnlineCourses\Udemy_nerf\dataset\fox"
    camera_intrinsic_train_data = data + "/train/intrinsics"
    camera_positional_train_data = data + "/train/pose"
    imgs_train_data = data + "/imgs/train"
    batch_size = 1024
    img_res_x = 400
    img_res_y = 400
    num_bins = 100
    device = "cuda"

    loader = dloader.DataSetLoader()
    train_imgs = loader.load_img_dataset(imgs_train_data)
    train_cam_intr = loader.load_cameras_intrinsics(camera_intrinsic_train_data)
    train_cam_poses = loader.load_cameras_positions(camera_positional_train_data)

    NerfRender = render.NerfRender(img_res_x, img_res_y, num_bins, device)
    train_rays_o, train_rays_d = NerfRender.generate_rays(train_cam_poses, train_cam_intr)
    train_target_px_vals = NerfRender.generate_target_pixel_arr(train_imgs)

