from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Activation, BatchNormalization, Conv2DTranspose, Concatenate
from UNET.Models.PartInterface import Part
class Massachusetts_Encoder(Part):
    def __init__(self, normalize_inputs):
        self.normalize_inputs = normalize_inputs
        self.build_model()
    def build_model(self):
        self.skip_connections = []

        x = Conv2D(64, (3, 3), padding='same', name='block1_conv1')(self.normalize_inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(64, (3, 3), padding='same', name='block1_conv2')(x)
        x = BatchNormalization()(x)
        block_1_out = Activation('relu')(x)
        self.skip_connections.append(block_1_out)
        x = MaxPooling2D()(block_1_out)

        x = Conv2D(128, (3, 3), padding='same', name='block2_conv1')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(128, (3, 3), padding='same', name='block2_conv2')(x)
        x = BatchNormalization()(x)
        block_2_out = Activation('relu')(x)
        self.skip_connections.append(block_2_out)
        x = MaxPooling2D()(block_2_out)

        x = Conv2D(256, (3, 3), padding='same', name='block3_conv1')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(256, (3, 3), padding='same', name='block3_conv2')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(256, (3, 3), padding='same', name='block3_conv3')(x)
        x = BatchNormalization()(x)
        block_3_out = Activation('relu')(x)
        self.skip_connections.append(block_3_out)
        x = MaxPooling2D()(block_3_out)

        x = Conv2D(512, (3, 3), padding='same', name='block4_conv1')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(512, (3, 3), padding='same', name='block4_conv2')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(512, (3, 3), padding='same', name='block4_conv3')(x)
        x = BatchNormalization()(x)
        block_4_out = Activation('relu')(x)
        self.skip_connections.append(block_4_out)




