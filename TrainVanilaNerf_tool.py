import Model.DataLoader as dloader



if __name__ == '__main__':
    data = r"D:\Myfiles\Projects\OnlineCourses\Udemy_nerf\dataset\fox"
    camera_intrinsic_train_data = data + "/train/intrinsics"
    camera_positional_train_data = data + "/train/pose"
    imgs_train_data = data + "/imgs/train"

    loader = dloader.DataSetLoader()
    loader.load_img_dataset(imgs_train_data)
