import os
import cv2
import numpy as np
from patchify import patchify
from pathlib import PurePath
from Dataset.DatasetUtils import DatasetUtils

class Patchify():

    def patch(self, color_paths_packs, gray_paths_packs, patch_size):

        for paths_pack_key in color_paths_packs:
            for path in color_paths_packs[paths_pack_key]:
                p = PurePath(path)
                spam = list(p.parts)
                spam.insert(4, 'patches')
                out_path = str(PurePath('').joinpath(*spam)).replace('\\', '/')
                self.extract_patches(path, out_path, patch_size, True)

        for paths_pack_key in gray_paths_packs:
            for path in gray_paths_packs[paths_pack_key]:
                p = PurePath(path)
                spam = list(p.parts)
                spam.insert(4, 'patches')
                out_path = str(PurePath('').joinpath(*spam)).replace('\\', '/')
                self.extract_patches(path, out_path, patch_size, True)

    def extract_patches(self, path, out_path, patch_size, isColorImage):
        print(out_path)
        file, file_extension = os.path.splitext(out_path)

        if isColorImage:
            img = cv2.imread(path)
            img = np.array(img)
            img = DatasetUtils.resize_3d_img(img, patch_size)
            patches = patchify(img, (patch_size, patch_size, 3), step=patch_size)
        else:
            img = cv2.imread(path, 0)
            img = np.array(img)
            img = DatasetUtils.resize_2d_img(img, patch_size)
            patches = patchify(img, (patch_size, patch_size), step=patch_size)

        patches = np.squeeze(patches)
        for i in range(len(patches[0])):
            for j in range(len(patches[1])):
                patch = patches[i, j, :, :]
                out_path = f"{file}_patch_{i}_{j}{file_extension}"
                cv2.imwrite(out_path, patch)


