import numpy as np
import imageio

from pathlib import Path
from natsort import natsorted


class DataSetLoader:
    """
    A class for loading input data for NeRF generation.
    """
    def __init__(self):
        self.supported_ext = ['.png', '.jpg', '.exr']

    @staticmethod
    def load_cameras_intrinsics(files_path: str):
        """
        A function for loading cameras intrinsic params stored in .txt files
        :param file_path: a string variable with path to the folder with .txt files
        :return: a numpy ndarray (n x 4 x 4) with intrinsic data of the each input camera.
        """

        files_path = Path(files_path)
        cam_intrinsics_files = natsorted([f for f in files_path.iterdir() if f.suffix.lower() == '.txt'], key=lambda y: y.as_posix().lower())
        files_number = len(cam_intrinsics_files)
        cam_intrinsics = np.zeros((files_number, 4, 4))

        for i, f in enumerate(cam_intrinsics_files):
            cam_intrinsic_data = open(f.as_posix()).read().split()
            cam_intrinsics[i] = np.array(cam_intrinsic_data, dtype=np.float32).reshape((4, 4))

        return cam_intrinsics

    @staticmethod
    def load_cameras_positions(files_path: str):
        """
        A function for loading cameras transformation data (rotations + translations) stored in .txt files
        :param file_path: a string variable with path to the folder with .txt files
        :return: a numpy ndarray (n x 4 x 4) with transforms for each input camera.
        """
        files_path = Path(files_path)
        cam_pose_files = natsorted([f for f in files_path.iterdir() if f.suffix.lower() == ".txt"], key=lambda y: y.as_posix().lower())
        files_number = len(cam_pose_files)
        cam_poses = np.zeros((files_number, 4, 4))

        for i, f in enumerate(cam_pose_files):
            cam_pose = open(f.as_posix()).read().split()
            cam_poses[i] = np.array(cam_pose, dtype=np.float32).reshape((4, 4))

        return cam_poses

    def load_img_dataset(self, files_path: str):
        """
        A function for loading images for NeRF training.
        :param files_path: a string variable with path to the folder with images. Check self.supported_ext for image types.
        :return: a numpy ndarray (N x img_w x img_h x num_ch) with loaded images.
        """
        files_path = Path(files_path)
        img_files = natsorted([f for f in files_path.iterdir() if f.suffix.lower() in self.supported_ext], key=lambda y: y.as_posix().lower())
        images = []

        assert len(img_files) > 0

        for i, f in enumerate(img_files):
            img = imageio.imread(f.as_posix())
            if max(img[:, 0].flatten()) == 255 or max(img[:, 1].flatten()) == 255 or max(img[:, 2].flatten()) == 255:
                img /= 255.0
            images.append(img[None, ...])

        images = np.concatenate(images)
        if images.shape[3] == 4:
            images = images[..., :3] * images[..., -1:] + (1 - images[..., -1:])

        return images
