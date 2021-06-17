import os
import cv2
import numpy as np

class DatasetUtils():

    @staticmethod
    def get_files_paths_pack(directory, file_extension):
        return sorted(
            [
                os.path.join(directory, filename)
                for filename in os.listdir(directory)
                if filename.endswith(file_extension)
            ]
        )

    @staticmethod
    def lists_to_numpy_array(lists):
        arrays = []
        for list in lists:
            arrays.append(np.array(list))
        return np.array(arrays)

    @staticmethod
    def resize_2d_img(img, patch_size):
        height, width = img.shape
        return cv2.resize(img, (width - width % patch_size, height - height % patch_size))

    @staticmethod
    def resize_3d_img(img, patch_size):
        height, width, c = img.shape
        return cv2.resize(img, (width - width % patch_size, height - height % patch_size))

    @staticmethod
    def preprocess_mask_image2(image, class_num, color_limit):
        pic = np.array(image)
        img = np.zeros((pic.shape[0], pic.shape[1], 1))
        np.place(img[:, :, 0], pic[:, :, 0] >= color_limit, 1)
        return img


