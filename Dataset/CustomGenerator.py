from abc import ABC, abstractmethod
from keras.utils import Sequence
import numpy as np

class CustomGenerator(ABC, Sequence):

    def __init__(self, image_filenames, labels_filenames, batch_size):
        self.image_filenames = image_filenames
        self.labels_filenames = labels_filenames
        self.batch_size = batch_size

    def __len__(self):
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)

    @abstractmethod
    def __getitem__(self, idx):
        pass