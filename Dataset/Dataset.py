from abc import ABC, abstractmethod
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

class Dataset(ABC):
    X_train_filenames = np.array([])
    X_val_filenames = np.array([])
    y_train_filenames = np.array([])
    y_val_filenames = np.array([])

    def to_grayscale(self, dirs_dict, paths_dict):
        for paths_key, dir_key in zip(paths_dict, dirs_dict):
            for path in paths_dict[paths_key]:
                img = cv2.imread(path, 0)
                cv2.imwrite(dirs_dict[dir_key] + os.path.basename(path), img)

    def split_dataset_filenames(self, images_paths_numpy, bin_labels_paths_numpy, test_size = 0.2, random_state = 1):
        return train_test_split(images_paths_numpy, bin_labels_paths_numpy, test_size=test_size, random_state=random_state)

    @abstractmethod
    def get_dataset_dirs(self):
        pass

    @abstractmethod
    def get_path_packs(self, dirs):
        pass

    @abstractmethod
    def get_filenames_to_grayscale(self, dirs):
        pass

    def grayscale(self, grayscale_dirs_paths_dict):
        color_filenames = self.get_path_packs(grayscale_dirs_paths_dict)
        self.to_grayscale(grayscale_dirs_paths_dict, color_filenames)
