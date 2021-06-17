import numpy as np
import cv2

from Dataset.CustomGenerator import CustomGenerator
from Dataset.DatasetUtils import DatasetUtils

class MassachusettsGenerator(CustomGenerator):

    def __getitem__(self, idx):

        batch_x = self.image_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.labels_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        images = []
        labels = []
        for filename in batch_x:
            img = cv2.imread(filename)
            img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_AREA)
            images.append(np.array(img).astype(np.float32) / 255)

        for filename in batch_y:
            mask = cv2.imread(str(filename))
            mask = cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_AREA)
            mask = DatasetUtils.preprocess_mask_image2(mask, 2, 50)
            labels.append(np.array(mask).astype(np.float32))

        return np.array(images), np.array(labels)