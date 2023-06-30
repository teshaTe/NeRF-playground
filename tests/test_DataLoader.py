import sys
import os
import numpy as np

root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_dir)

from pathlib import Path
import Utils.DataLoader as dataloader

test_intrinsic_data = Path(root_dir + "/tests/test_data/intrinsics")
test_transforms_data = Path(root_dir + "/tests/test_data/pose")
test_images_data = Path(root_dir + "/tests/test_data/imgs")
data_loader = dataloader.DataSetLoader()


def test_camera_intrinsic_loading():
    intrinsic_matrix = np.array([[1333.3333333333335, 0.0, 200, 0.0],
                                 [0.0, 1333.3333333333335, 200, 0.0],
                                 [0.0, 0.0, 1.0, 0.0],
                                 [0.0, 0.0, 0.0, 1.0]])
    test_intrinsics = data_loader.load_cameras_intrinsics(test_intrinsic_data.as_posix())
    for m in test_intrinsics:
        assert m.all() == intrinsic_matrix.all()


def test_camera_transforms_loading():
    cameras_tr0 = np.array([[0.6859206557273865, -0.32401347160339355, 0.6515582203865051, 7.358891487121582],
                            [0.7276763319969177, 0.305420845746994, -0.6141703724861145, -6.925790786743164],
                            [0.0, 0.8953956365585327,  0.44527140259742737, 4.958309173583984],
                            [0.0, 0.0, 0.0, 1.0]])
    cameras_tr1 = np.array([[-4.371138828673793e-08, -0.9950000047683716, 0.09987492859363556, 0.9987491965293884],
                            [1.0, -4.349283244664548e-08, 4.365671824047013e-09, 0.0],
                            [0.09987492859363556, 0.9950000047683716, 9.949999809265137, 0.0],
                            [0.0, 0.0, 0.0, 1.0]])
    cameras_tr2 = np.array([[0.6754904389381409, 0.7263082265853882, -0.12723617255687714, -1.2723619937896729],
                            [-0.7373687624931335, 0.6653580665588379, -0.11655880510807037, -1.1655877828598022],
                            [0.0, 0.17255434393882751, 0.9850000143051147, 9.850000381469727],
                            [0.0, 0.0, 0.0, 1.0]])
    cameras_tr = np.array([cameras_tr0, cameras_tr1, cameras_tr2])

    test_transforms = data_loader.load_cameras_positions(test_transforms_data.as_posix())

    for r, t in zip(cameras_tr, test_transforms):
        assert r.all() == t.all()


def test_image_loading():
    test_images = data_loader.load_img_dataset(test_images_data.as_posix())
    assert len(test_images) == 2
    assert test_images[0].shape == (400, 400, 3)
    for i in test_images:
        assert i is not None
        assert i.all() != 1 or i.all() != 255 or i.all() != 0

    assert test_images[0].all() == test_images[1].all()
