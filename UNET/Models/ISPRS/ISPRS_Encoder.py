from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Activation, BatchNormalization, Conv2DTranspose, Concatenate, advanced_activations
from UNET.Models.PartInterface import Part

class ISPRS_Encoder(Part):
    def __init__(self, normalize_inputs):
        self.normalize_inputs = normalize_inputs
        self.build_model()
    def build_model(self):
        self.skip_connections = []
        # Block 1
        conv1 = Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same')(self.normalize_inputs)
        conv1 = BatchNormalization(axis=1)(conv1)
        conv1 = advanced_activations.ELU()(conv1)
        conv1 = Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same')(conv1)
        conv1 = BatchNormalization(axis=1)(conv1)
        block_1_out = advanced_activations.ELU()(conv1)
        self.skip_connections.append(block_1_out)
        pool1 = MaxPooling2D((2, 2))(block_1_out)
        # Block 2
        conv2 = Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same')(pool1)
        conv2 = BatchNormalization(axis=1)(conv2)
        conv2 = advanced_activations.ELU()(conv2)
        conv2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv2)
        conv2 = BatchNormalization(axis=1)(conv2)
        block_2_out = advanced_activations.ELU()(conv2)
        self.skip_connections.append(block_2_out)
        pool2 = MaxPooling2D((2, 2))(block_2_out)
        # Block 3
        conv3 = Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same')(pool2)
        conv3 = BatchNormalization(axis=1)(conv3)
        conv3 = advanced_activations.ELU()(conv3)
        conv3 = Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same')(conv3)
        conv3 = BatchNormalization(axis=1)(conv3)
        block_3_out = advanced_activations.ELU()(conv3)
        self.skip_connections.append(block_3_out)
        pool3 = MaxPooling2D((2, 2))(block_3_out)
        # Block 4
        conv4 = Conv2D(256, (3, 3), kernel_initializer='he_normal', padding='same')(pool3)
        conv4 = BatchNormalization(axis=1)(conv4)
        conv4 = advanced_activations.ELU()(conv4)
        conv4 = Conv2D(256, (3, 3), kernel_initializer='he_normal', padding='same')(conv4)
        conv4 = BatchNormalization(axis=1)(conv4)
        block_4_out = advanced_activations.ELU()(conv4)
        self.skip_connections.append(block_4_out)
        pool4 = MaxPooling2D((2, 2))(block_4_out)
        # Block 5
        conv5 = Conv2D(512, (3, 3), kernel_initializer='he_normal', padding='same')(pool4)
        conv5 = BatchNormalization(axis=1)(conv5)
        conv5 = advanced_activations.ELU()(conv5)
        conv5 = Conv2D(512, (3, 3), kernel_initializer='he_normal', padding='same')(conv5)
        conv5 = BatchNormalization(axis=1)(conv5)
        block_5_out = advanced_activations.ELU()(conv5)
        self.skip_connections.append(block_5_out)