import torch
import torch.nn as nn

from torch.cuda.amp import custom_fwd, custom_bwd


class TruncExp(torch.autograd.Function):
    """
    A class that defines a custom activation function - truncated exponential.
    """
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


class VanillaNerfModel(nn.Module):
    """
     A vanilla NeRF model.
    """
    def __init__(self, L_pos=10, L_dir=4, hidden_dim=256):
        """
        A constructor for the current model. L is a power parameter in the position encoding eq (4) in the vanila NeRF
        :param L_pos: int parameter for the position encoding
        :param L_dir: int parameter for the direction encoding
        :param hidden_dim: int parameter that defines the size of every hidden layer
        """
        super().__init__()

        # positional encoding parameters for position and direction vectors
        # check eq (4) in the vanila NeRF paper
        self.L_pos = L_pos
        self.L_dir = L_dir

        # positional encoding eq (4): 2 -> for each value we need to compute sin(2^(L-1) * p) and cos(2^(L-1) * p);
        #                             3 -> each pos is 3 coords
        #                            +3 -> the position is kept - important to take it into account as sometimes
        #                                  it can be omitted in the papers with NeRF implementation

        # architecture of the network, fig. 7 in vanila NeRF paper
        # block1 -> 5 layers before gamma(x) (positional encoding of the input pos) is added
        self.block1 = nn.Sequential(nn.Linear(L_pos * 2 * 3 + 3, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU())

        # block2 -> 4 layers before gamma(d) (positional encoding of the direction) is added + density
        self.block2 = nn.Sequential(nn.Linear(hidden_dim + L_pos * 2 * 3 + 3, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim + 1))

        # block3 -> 2 last layers, with last layer which outputs rgb value
        self.rgb_block3 = nn.Sequential(nn.Linear(hidden_dim + L_dir * 2 * 3 + 3, hidden_dim // 2), nn.ReLU(),
                                        nn.Linear(hidden_dim // 2, 3), nn.Sigmoid())

    def positional_encoding(self, x, L):
        """
        A function for positional encoding
        :param x: a torch tensor with position vector (nx3) data
        :param L: an integer parameter defining the number of features to be used
        :return: torch tensor with encoded input position
        """
        output = [x]
        # iterate through the features
        for j in range(L):
            # pi is omitted here as it cause an issue -> https://github.com/bmild/nerf/issues/12
            output.append(torch.sin(2 ** j * x))
            output.append(torch.cos(2 ** j * x))
        return torch.cat(output, dim=1)

    def forward(self, pos, dirs):
        """
        A function for computing the colour which will be used for computing the result of the volume rendering eq.
        :param pos: a torch tensor with positional data
        :param dirs: a torch tensor with direction data
        :return: a torch tensor with estimated colours and a torch tensor with estimated density
        """

        # computing positional encoding for the input positions & directions
        x_emb = self.positional_encoding(pos, self.L_pos)  # shape: batch_size x (L_pos*2*3 + 3)
        d_emb = self.positional_encoding(dirs, self.L_dir)  # shape: batch_size x (L_dir*2*3 + 3)

        # passing the input data through neural network layers
        block = self.block1(x_emb)                             # shape: batch_size x hidden_dim
        block = self.block2(torch.cat((block, x_emb), dim=1))  # shape: batch_size x hidden_dim+1
        sigma = block[:, -1]                                   # last dimension is sigma
        block = block[:, :-1]                                  # back to 256 neurons instead of 257
        color = self.rgb_block3(torch.cat((block, d_emb), dim=1))

        return color, torch.relu(sigma)


class VanillaNerfVoxelModel(nn.Module):
    """
    A vanilla NeRF model computed on voxel grid
    """
    def __init__(self, grid_res, grid_scale=1, device='cpu'):
        """
        A constructor for the current model.
        :param grid_res: an integer parameter that defines the size of the regular voxel grid
        :param grid_scale: a float parameter that controls the scale of the creating voxel grid
        :param device: the str parameter with the name of the device -> 'cpu' or 'cuda'
        """
        super().__init__()

        if device == 'cpu' or device == 'cuda':
            self.device = device
        else:
            raise ValueError("Unknown device has been specified!")

        self.grid_res = grid_res
        self.scale = grid_scale
        self.voxel_grid = nn.Parameter(torch.rand((grid_res, grid_res, grid_res, 4), device=self.device, requires_grad=True))

    def forward(self, pos):
        """
        A function for computing the colour which will be used for computing the result of the volume rendering eq.
        :param pos: a torch tensor with positional data
        :return: a torch tensor with estimated colours and a torch tensor with estimated density
        """
        x = pos[:, 0]
        y = pos[:, 1]
        z = pos[:, 2]

        # condition that defines the position of the voxel grid in space: -0.5 - +0.5 along y, y, z
        cond = (x.abs() < self.scale / 2) & (y.abs() < self.scale / 2) & (z.abs() < self.scale / 2)
        cond = cond.to(self.device)

        # setting up initial values
        colors_and_densities = torch.zeros((pos.shape[0], 4), device=self.device)

        # defining indices in the voxel grid
        indx = (x[cond] / (self.scale / self.grid_res) + self.grid_res / 2).type(torch.long)
        indy = (y[cond] / (self.scale / self.grid_res) + self.grid_res / 2).type(torch.long)
        indz = (z[cond] / (self.scale / self.grid_res) + self.grid_res / 2).type(torch.long)

        colors_and_densities[cond, :3] = self.voxel_grid[indx, indy, indz, :3]
        colors_and_densities[cond, -1] = self.voxel_grid[indx, indy, indz, -1]

        return torch.sigmoid(colors_and_densities[:, :3]), torch.relu(colors_and_densities[:, -1])
