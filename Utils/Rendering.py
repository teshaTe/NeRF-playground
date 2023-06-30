import numpy as np
import torch
import tqdm


class NerfRender:
    """
    This class provides functionality for rendering NeRFs.
    The comments in the code refers to the vanila NeRF paper:
    https://www.matthewtancik.com/nerf
    """
    def __init__(self, res_x, res_y, num_bins=100, device='cpu'):
        """
        A constructor of the class for rendering NeRF model.
        :param res_x: an int parameter that corresponds to the width of the image being used in training/testing
        :param res_y: an int parameter that corresponds to the height of the image being used in training/testing
        :param num_bins: an integer parameter specifying the amount of bins for sampling t values
        :param device:
        """
        self.res_x = res_x
        self.res_y = res_y
        self.num_bins = num_bins
        if device == 'cpu' or device == 'cuda':
            self.device = device
        else:
            raise ValueError("Unknown device has been specified!")

    def generate_rays(self, cams_poses, cams_intrinsics):
        """
        A function for generation of the camera rays: origins and directions arrays
        :param cams_poses: a numpy ndarray (N x 4 x 4) with cameras' transforms (rotation matrix R and translation t vector);
                           R   | t  where R 3x3 matrix and t 1x3 vector
                           0 0 0 1
        :param cams_intrinsics: a numpy ndarray (N x 3 x 3) with intrinsic parameters of the cameras;
                                fx 0 cx 0  where fx, fy - focal length of the camera in pixels along x and y axis;
                                0 fy cy 0        cx, cy - coordinates of the principal point of the camera plane.
                                0  0  1 0
        :return: torch tensor with positional data of the camera rays origins (rays_o);
                 torch tensor with directional data of the camera rays (rays_d)
        """

        # initializing the arrays
        total = len(cams_poses)
        # cams_poses = torch.from_numpy(cams_poses).type(torch.float)
        # cams_intrinsics = torch.from_numpy(cams_intrinsics).type(torch.float)
        rays_o = np.zeros((total, self.res_x * self.res_y, 3))
        rays_d = np.zeros((total, self.res_x * self.res_y, 3))

        # assembling rays arrays: origin & direction data per image
        for i in range(total):
            # defining camera plane
            u = np.arange(self.res_x)
            v = np.arange(self.res_y)
            u, v = np.meshgrid(u, v)  # image plane

            # intrinsics, extracting focal length in pixels along x and y axis (image plane)
            # Intrinsic matrix (https://towardsdatascience.com/what-are-intrinsic-and-extrinsic-camera-parameters-in-computer-vision-7071b72fb8ec):

            fx = cams_intrinsics[i][0, 0]
            fy = cams_intrinsics[i][1, 1]

            # assembling direction vectors per image
            rays_dirs = np.stack(((u - self.res_x / 2) / fx, (-(v - self.res_y / 2) / fy), -np.ones_like(u)), axis=-1)
            rays_dirs = np.matmul(cams_poses[i][:3, :3], rays_dirs[..., None]).squeeze(-1)

            # normalizaton of the direction vectors
            rays_dirs = rays_dirs / np.linalg.norm(rays_dirs, axis=-1, keepdims=True)

            rays_d[i] = rays_dirs.reshape(-1, 3)
            rays_o[i] += cams_poses[i][:3, 3]

        return torch.from_numpy(rays_o).to(self.device).type(torch.float), torch.from_numpy(rays_d).to(self.device).type(torch.float)

    def generate_target_pixel_arr(self, images):
        """
        A function for conversion of the input array with images to pixell arrays
        :param images: a numpy ndarray (N x W x H x 3) with imagery data
        :return: a numpy ndarray with pixel data
        """
        return torch.from_numpy(images.reshape((len(images), self.res_x*self.res_y, 3))).type(torch.float).to(self.device)

    def compute_accumulated_transmittance(self, betas: torch.FloatTensor):
        """
        A function for computing the accumulated transmittance along the ray. It defines the probability
        that the ray travels from tn to t without hitting any other particle.
        :param betas: a torch tensor with density values along the ray
        :return: a torch tensor with transmittance values
        """
        accum_transmittance = torch.cumprod(betas, 1)
        return torch.cat((torch.ones(accum_transmittance.shape[0], 1, device=self.device),
                          accum_transmittance[:, :-1]), dim=1)

    def render_view(self, model, rays_o: torch.FloatTensor, rays_dirs: torch.FloatTensor, tn, tf, white_background=False):
        """
        A function for rendering the specified view using the trained NeRF model.
        :param model: a torch.nn.Module object with trained model
        :param rays_o: a torch tensor with coordinates of the camera rays origins for the current view
        :param rays_dirs: a torch tensor with direction vectors of the camera rays for the current view
        :param tn: the lower bound of the point sampling interval (see eq (1) in NeRF paper)
        :param tf: the upper bound of the point sampling interval (see eq (1) in NeRF paper)
        :param white_background: a boolean flag that enables/disables background noise reduction for synthetic data
        :return: a torch tensor with colour of the NeRF object - solution of the volume rendering equation
        """

        rays_o = rays_o.to(self.device)
        rays_dirs = rays_dirs.to(self.device)

        # Partitioning of the interval [tn, tf] into N evenly-spaced bins
        # shape of t array: (bins_num, )
        t = torch.linspace(tn, tf, self.num_bins).to(self.device)
        t_inf = torch.tensor([1e10]).to(self.device)

        # Computing the distance between adjacent samples (the last value in deltas should be inf value)
        # shape of deltas array: (bins_num, )
        deltas = torch.cat((t[1:] - t[:-1], t_inf))

        # computing the position of the ray at sample t: x(t) = rays_o + t * rays_d
        # shape of rays_o & rays_dirs: (rays_num, 3)
        # we want x to have the shape: (rays_num, bins_num, 3) -> t.reshape((1, bins_num, 1)), rays_o.reshape((rays_num, 1, 3))
        x = rays_o.unsqueeze(1) + t.unsqueeze(0).unsqueeze(-1) * rays_dirs.unsqueeze(1)

        # computing rays intersection with the object, getting c_i from eq (3)
        x_model = x.reshape(-1, 3)
        rays_dirs_model = rays_dirs.expand(x.shape[1], x.shape[0], 3).transpose(0, 1).reshape(-1, 3)
        colors, density = model.forward(x_model, rays_dirs_model)

        # reshaping the data
        colors = colors.reshape((x.shape[0], self.num_bins, 3))
        density = density.reshape((x.shape[0], self.num_bins))

        # Volume rendering equation quadrature rule, eq (3): C(x) = Sum from i=1 to N (T_i * alpha_i * c_i),
        # where alpha_i = 1 - exp(-sigma_i * delta_i); here sigma_i == density
        alpha = 1 - torch.exp(-density * deltas.unsqueeze(0))

        # transmittance, eq (3): T = exp(- Sum from j=1 to i-1 (sigma_j * delta_j))
        T = self.compute_accumulated_transmittance(1 - alpha)

        # weights for the estimated colors
        weights = T * alpha

        # Adding regularization term for handling noise in the background (FOR SYNTHETIC DATA)
        if white_background:
            c = (weights.unsqueeze(-1) * colors).sum(1)  # summing across first dim,  shape: num_rays, 3
            weights_sum = weights.sum(-1)                # summing across last dim, shape: num_rays.shape
            c = c + 1 - weights_sum.unsqueeze(-1)        # empty space -> convert it to white
        else:
            c = (weights.unsqueeze(-1) * colors).sum(1)

        return c

    @torch.no_grad()
    def generate_view(self, model, rays_o: torch.FloatTensor, rays_dirs: torch.FloatTensor, tn, tf, chunk_size=10):
        """
        A function for rendering a novel view using trained NeRF model
        :param model: a torch.nn.Module object with trained model
        :param rays_o: a torch tensor with coordinates of the camera rays origins for the current view
        :param rays_dirs: a torch tensor with direction vectors of the camera rays for the current view
        :param tn: the lower bound of the point sampling interval (see eq (1) in NeRF paper)
        :param tf: the upper bound of the point sampling interval (see eq (1) in NeRF paper)
        :param chunk_size: an integer value that specifies the amount of chunks that rays_o and rays_dir will be split into
        :return: an image of the novel view
        """

        rays_o = rays_o.to(self.device)
        rays_dirs = rays_dirs.to(self.device)

        # subdividing input arrays to chunks to avoid out of memory error
        rays_o = rays_o.chunk(chunk_size)
        rays_dir = rays_dirs.chunk(chunk_size)

        # rendering a novel view
        image = []
        for o_batch, dir_batch, i in zip(rays_o, rays_dir, tqdm.trange(len(rays_o))):
            img_batch = self.render_view(model, o_batch, dir_batch, tn, tf, white_background=True)
            image.append(img_batch)  # shape: N x 3 per img_batch

        image = torch.cat(image)
        image = image.reshape(self.res_y, self.res_x, 3).cpu().detach().numpy()

        return image
