from tensorflow.python.keras.layers import Input, Conv2D, MaxPooling2D, Activation, BatchNormalization, Conv2DTranspose, Concatenate
from UNET.Models.PartInterface import Part
class Massachusetts_Decoder(Part):
    def __init__(self, skip_connections, num_classes):
        self.skip_connections = skip_connections[::-1]
        self.num_classes = num_classes
        self.build_model()

    def build_model(self):
        x = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same', name='Conv2DTranspose_UP2')(self.skip_connections[0])
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Concatenate()([x, self.skip_connections[1]])
        x = Conv2D(256, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(256, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        # UP 3
        x = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', name='Conv2DTranspose_UP3')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Concatenate()([x, self.skip_connections[2]])
        x = Conv2D(128, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(128, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        # UP 4
        x = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', name='Conv2DTranspose_UP4')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Concatenate()([x, self.skip_connections[3]])
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        self.output = Activation('relu')(x)