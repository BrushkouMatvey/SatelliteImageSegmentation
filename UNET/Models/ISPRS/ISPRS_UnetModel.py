import tensorflow as tf

from UNET.Models.ISPRS.ISPRS_Decoder import ISPRS_Decoder
from UNET.Models.ISPRS.ISPRS_Encoder import ISPRS_Encoder
from UNET.Models.ModelsUtils import ModelsUtils


class ISPRS_UnetModel():

    def __init__(self, num_classes=1, input_shape=(512, 512, 1)):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.build_model()

    def build_model(self):

        # Build the model
        inputs = tf.keras.layers.Input(self.input_shape)
        s = inputs

        encoder = ISPRS_Encoder(s)
        decoder = ISPRS_Decoder(encoder.skip_connections, 1)

        outputs = tf.keras.layers.Conv2D(self.num_classes, (1, 1), activation='sigmoid', padding='same')(decoder.output)
        self.model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

        self.model.compile(optimizer=tf.keras.optimizers.Nadam(lr=1e-3), loss=ModelsUtils.jaccard_coef_loss,
                      metrics=['binary_crossentropy', ModelsUtils.jaccard_coef_int])
        self.model.summary()

        return self.model