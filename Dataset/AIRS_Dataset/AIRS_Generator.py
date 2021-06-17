from abc import ABC

from keras.utils import to_categorical
from keras.utils import normalize, Sequence
import numpy as np
import cv2

from Dataset.CustomGenerator import CustomGenerator


class AIRS_Generator(CustomGenerator):

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.labels_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]

        images = np.expand_dims(normalize(np.array([cv2.imread(str(file_name), 0) for file_name in batch_x]), axis=1),
                                3)
        labels = np.expand_dims(np.array([cv2.imread(str(file_name), 0) for file_name in batch_y]), 3)
        return images, labels