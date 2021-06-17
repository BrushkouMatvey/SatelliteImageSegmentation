import numpy as np
import cv2
import os
from Dataset.Dataset import Dataset
from Dataset.DatasetUtils import DatasetUtils
from Dataset.Patchify import Patchify
from UNET.UnetParams import UnetParams


class AIRS_Dataset(Dataset):

    def __init__(self):
        self.color_dirs_paths_dict, self.grayscale_dirs_paths_dict, self.patch_dirs_paths_dict, self.only_buildings_dirs_paths_dict = self.get_dataset_dirs()
        self.unet_params = UnetParams(1024, 8, 1)
        # self.grayscale(self.grayscale_dirs_paths_dict)
        # self.to_patches()
        # self.save_patches_with_buildings()
        self.X_train_filenames, self.X_val_filenames, self.y_train_filenames, self.y_val_filenames = self.get_learn_filenames()

    #grayscale
    def get_filenames_to_grayscale(self, color_dirs_paths_dict):
        return self.get_path_packs(color_dirs_paths_dict)

    #to_pathces
    def to_patches(self):
        color_paths_packs, gray_paths_packs = self.get_path_packs(self.color_dirs_paths_dict), self.get_path_packs(self.grayscale_dirs_paths_dict)
        patchify = Patchify()
        patchify.patch(color_paths_packs, gray_paths_packs, 1024)

    def get_dataset_dirs(self):
        color_paths_dict = {
            'image': '../Data/AIRS/test/color/image/',
            'bin': '../Data/AIRS/test/color/label/bin/',
            'vis': '../Data/AIRS/test/color/label/vis/'
        }
        grayscale_paths_dict = {
            'image': '../Data/AIRS/test/grayscale/image/',
            'bin': '../Data/AIRS/test/grayscale/label/bin/',
            'vis': '../Data/AIRS/test/grayscale/label/vis/'
        }
        patch_dirs_paths_dict = {
            'image': '../Data/AIRS/test/patches/grayscale/image/',
            'bin': '../Data/AIRS/test/patches/grayscale/label/bin/',
            'vis': '../Data/AIRS/test/patches/grayscale/label/vis/'
        }
        only_buildings_dirs_paths_dict = {
            'image': '../Data/AIRS/test/patches/grayscale_only_buildings/image/',
            'bin': '../Data/AIRS/test/patches/grayscale_only_buildings/label/bin/',
            'vis': '../Data/AIRS/test/patches/grayscale_only_buildings/label/vis/'
        }

        return color_paths_dict, grayscale_paths_dict, patch_dirs_paths_dict, only_buildings_dirs_paths_dict

    def get_path_packs(self, dirs):
        path_pack = {
            'image': DatasetUtils.get_files_paths_pack(dirs["image"], '.tif'),
            'vis': DatasetUtils.get_files_paths_pack(dirs["vis"], '.tif'),
            'bin': DatasetUtils.get_files_paths_pack(dirs["bin"], '.tif')
        }
        return path_pack

    def get_learn_filenames(self):
        self.path_pack = self.get_path_packs(self.only_buildings_dirs_paths_dict)

        # DatasetUtils.lists_to_numpy_array([self.path_pack["image"], self.path_pack["vis"], self.path_pack["bin"]])

        self.path_pack["image"] = np.array(self.path_pack["image"])
        self.path_pack["vis"] = np.array(self.path_pack["vis"])
        self.path_pack["bin"] = np.array(self.path_pack["bin"])

        return self.split_dataset_filenames(self.path_pack["image"], self.path_pack["vis"])

    def save_patches_with_buildings(self):
        color_paths_packs, gray_paths_packs, patch_paths_packs = self.get_path_packs(self.color_dirs_paths_dict), self.get_path_packs(self.grayscale_dirs_paths_dict), self.get_path_packs(self.patch_dirs_paths_dict)

        paths_vis_with_buildings = []
        for path in patch_paths_packs['vis']:
            img = cv2.imread(path, 0)
            number_of_white_pix = np.sum(img == 255)
            if number_of_white_pix > 0:
                filename = os.path.basename(path)
                cv2.imwrite(f'{self.only_buildings_dirs_paths_dict["vis"]}label/vis/{filename}', img)
                paths_vis_with_buildings.append(path)

        paths_bin_with_buildings = []
        paths_images_with_buildings = []
        for path in patch_paths_packs['bin']:
            img = cv2.imread(path, 0)
            number_of_white_pix = np.sum(img == 1)
            if number_of_white_pix > 0:
                filename = os.path.basename(path)
                cv2.imwrite(f'{self.only_buildings_dirs_paths_dict["bin"]}{filename}', img)
                grayscale_img = cv2.imread(f'{self.grayscale_dirs_paths_dict["image"]}{filename}', 0)
                cv2.imwrite(f'{self.only_buildings_dirs_paths_dict["image"]}{filename}', grayscale_img)
                paths_bin_with_buildings.append(path)
                paths_images_with_buildings.append(f'{self.only_buildings_dirs_paths_dict["image"]}{filename}')

        print(len(paths_vis_with_buildings))
        print(len(paths_bin_with_buildings))
        print(len(paths_images_with_buildings))
        np.save('vis.npy', paths_vis_with_buildings)
        np.save('bin.npy', paths_bin_with_buildings)
        np.save('images.npy', paths_images_with_buildings)

