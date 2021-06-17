from tensorflow.python.keras.layers import Conv2D, BatchNormalization, Concatenate, advanced_activations, UpSampling2D
from UNET.Models.PartInterface import Part

class ISPRS_Decoder(Part):

    def __init__(self, skip_connections, num_classes):
        self.skip_connections = skip_connections[::-1]
        self.num_classes = num_classes
        self.build_model()

    def build_model(self):
        up = UpSampling2D((2, 2))(self.skip_connections[0])
        conv6 = Concatenate()([up, self.skip_connections[1]])
        conv6 = Conv2D(256, (3, 3), kernel_initializer='he_normal', padding='same')(conv6)
        conv6 = BatchNormalization(axis=1)(conv6)
        conv6 = advanced_activations.ELU()(conv6)
        conv6 = Conv2D(256, (3, 3), kernel_initializer='he_normal', padding='same')(conv6)
        conv6 = BatchNormalization(axis=1)(conv6)
        conv6 = advanced_activations.ELU()(conv6)

        up = UpSampling2D((2, 2))(conv6)
        conv7 = Concatenate()([up, self.skip_connections[2]])
        conv7 = Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same')(conv7)
        conv7 = BatchNormalization(axis=1)(conv7)
        conv7 = advanced_activations.ELU()(conv7)
        conv7 = Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same')(conv7)
        conv7 = BatchNormalization(axis=1)(conv7)
        conv7 = advanced_activations.ELU()(conv7)

        up = UpSampling2D((2, 2))(conv7)
        conv8 = Concatenate()([up, self.skip_connections[3]])
        conv8 = Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same')(conv8)
        conv8 = BatchNormalization(axis=1)(conv8)
        conv8 = advanced_activations.ELU()(conv8)
        conv8 = Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same')(conv8)
        conv8 = BatchNormalization(axis=1)(conv8)
        conv8 = advanced_activations.ELU()(conv8)

        up = UpSampling2D((2, 2))(conv8)
        conv9 = Concatenate()([up, self.skip_connections[4]])
        conv9 = Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same')(conv9)
        conv9 = BatchNormalization(axis=1)(conv9)
        conv9 = advanced_activations.ELU()(conv9)
        conv9 = Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same')(conv9)
        conv9 = BatchNormalization(axis=1)(conv9)
        conv9 = advanced_activations.ELU()(conv9)
        self.output = Conv2D(2, (3, 3), kernel_initializer='he_normal', padding='same')(conv9)















