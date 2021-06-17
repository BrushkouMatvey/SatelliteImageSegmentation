from Dataset.Dataset import Dataset
from Dataset.DatasetUtils import DatasetUtils
from UNET.UnetParams import UnetParams
import numpy as np
class MassachusettsDataset(Dataset):

    def __init__(self):
        self.train_dirs_dict, self.validation_dirs_dict = self.get_dataset_dirs()
        self.unet_params = UnetParams(1024, 2, 3)
        self.X_train_filenames, self.X_val_filenames, self.y_train_filenames, self.y_val_filenames = self.get_learn_filenames()

    def get_filenames_to_grayscale(self, color_dirs_paths_dict):
        return self.get_path_packs(color_dirs_paths_dict)

    def get_dataset_dirs(self):
        train_dirs_dict = {
            'image': '../Data/MassachusettsRoads/train/',
            'label': '../Data/MassachusettsRoads/train_labels/'
        }

        validation_dirs_dict = {
            'image': '../Data/MassachusettsRoads/val/',
            'label': '../Data/MassachusettsRoads/val_labels/'
        }

        return train_dirs_dict, validation_dirs_dict

    def get_path_packs(self, dirs):
        pass

    def get_learn_filenames(self):

        x_train_filenames = np.array(DatasetUtils.get_files_paths_pack(self.train_dirs_dict["image"], '.tiff'))
        y_train_filenames = np.array(DatasetUtils.get_files_paths_pack(self.train_dirs_dict["label"], '.tif'))

        x_val_filenames = np.array(DatasetUtils.get_files_paths_pack(self.validation_dirs_dict["image"], '.tiff'))
        y_val_filenames = np.array(DatasetUtils.get_files_paths_pack(self.validation_dirs_dict["label"], '.tif'))

        return x_train_filenames, x_val_filenames, y_train_filenames, y_val_filenames

