import torch
import torch.nn as nn
import tinycudann as tcnn
from torch.cuda.amp import custom_fwd, custom_bwd


class FFNerfModel(nn.Module):
    """
     A fully fused MLP NeRF model.
    """
    def __init__(self, L_pos=10, L_dir=4, hidden_dim=128):
        """
        A constructor for the current model. L is a power parameter in the position encoding eq (4) in the vanila NeRF
        :param L_pos: int parameter for the position encoding; check eq (4) in the vanila NeRF paper
        :param L_dir: int parameter for the direction encoding; check eq (4) in the vanila NeRF paper
        :param hidden_dim: int parameter that defines the size of every hidden layer
        """

        super().__init__()
        self.L_pos = L_pos
        self.L_dir = L_dir

        if hidden_dim > 128:
            hidden_dim = 128
            print("[WARNING] Currently tiny-cudann supports layeers witn sizes: 16, 32, 64, 128.")
            print("[WARNING] Falls back to 128.")

        # positional encoding eq (4): 2 -> for each value we need to compute sin(2^(L-1) * p) and cos(2^(L-1) * p);
        #                             3 -> each pos is 3 coords
        #                            +3 -> the position is kept - important to take it into account as sometimes
        #                                  it can be omitted in the papers with NeRF implementation

        # architecture of the network, fig. 7 in vanila NeRF paper
        # block1 -> 5 layers before gamma(x) (positional encoding of the input pos) is added
        encoding1 = {"otype": "Frequency",
                     "n_frequencies": L_pos}

        network1 = {"otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "ReLU",
                    "n_neurons": hidden_dim,
                    "n_hidden_layers": 4}

        self.block1 = tcnn.Network(L_pos * 2 * 3 + 3, hidden_dim, network1)
        # self.block1 = tcnn.NetworkWithInputEncoding(L_pos * 2 * 3 + 3, hidden_dim, encoding1, network1)

        # block2 -> 4 layers before gamma(d) (positional encoding of the direction) is added + density
        network2 = {"otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": hidden_dim,
                    "n_hidden_layers": 3}
        self.block2 = tcnn.Network(hidden_dim + L_pos * 2 * 3 + 3, hidden_dim+1, network2)
        # self.block2 = tcnn.NetworkWithInputEncoding(hidden_dim + L_pos * 2 * 3 + 3, hidden_dim+1, encoding1, network2)

        # block3 -> 2 last layers, with last layer which outputs rgb value
        encoding2 = {"otype": "Frequency",
                     "n_frequencies": L_dir}

        network3 = {"otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "Sigmoid",
                    "n_neurons": hidden_dim // 2,
                    "n_hidden_layers": 1}

        self.rgb_block3 = tcnn.Network(hidden_dim + L_dir * 2 * 3 + 3, 3, network3)
        # self.rgb_block3 = tcnn.NetworkWithInputEncoding(hidden_dim + L_dir * 2 * 3 + 3, 3, encoding2, network3)

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
