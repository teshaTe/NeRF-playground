import os
import torch
from pathlib import Path

import Model.VanillaModel as NerfModel
import Model.FullyFusedModel as NerfFFModel
import Utils.MeshExtraction as meshing


if __name__ == '__main__':
    root = Path(os.path.realpath(__file__)).parent

    device = "cuda"
    grid_res = 100
    grid_scale = 1.5
    mcube_thres = 3.0
    
    nerf_model = "FullyFusedMLP"  # Vanilla, FullyFusedMLP
    model_name = "nerf_model_v2"
    model_dir = root / "torch_models"

    if nerf_model == "Vanilla":
        model = NerfModel.VanillaNerfModel().to(device)
    elif nerf_model == "FullyFusedMLP":
        model = NerfFFModel.FFNerfModel().to(device)
    else:
        raise ValueError("Unknown model name has been specified!")

    model.load_state_dict(torch.load(model_dir.as_posix() + "/" + model_name + ".pt"))

    meshingsys = meshing.SimpleMeshNeRF(device)
    vertices, faces = meshingsys.generate_mesh(model, grid_res, grid_scale, mcube_thres)
    meshingsys.render_mesh(vertices, faces)
