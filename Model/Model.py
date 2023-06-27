import torch
import torch.nn as nn

from torch.cuda.amp import custom_fwd, custom_bwd


class TruncExp(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, dL_dout):
        x = ctx.saved_tensors[0]
        return dL_dout * torch.exp(x.clamp(-15, 15))


class VanilaNerfModel(nn.Module):
    def __init__(self):
        pass

    def positional_encoding(self):
        pass

    def forward(self):
        pass


class VanilaNerfVoxelModel(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass


