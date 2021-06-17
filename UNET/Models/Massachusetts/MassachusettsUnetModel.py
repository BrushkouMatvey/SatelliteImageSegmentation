import tensorflow as tf

from UNET.Models.Massachusetts.Massachusetts_Decoder import Massachusetts_Decoder
from UNET.Models.Massachusetts.Massachusetts_Encoder import Massachusetts_Encoder
from UNET.Models.ModelsUtils import ModelsUtils


class MassachusettsUnetModel():

    def __init__(self, num_classes=1, input_shape=(1024, 1024, 3)):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.build_model()

    def build_model(self):

        # Build the model
        inputs = tf.keras.layers.Input(self.input_shape)
        s = inputs

        encoder = Massachusetts_Encoder(s)
        decoder = Massachusetts_Decoder(encoder.skip_connections, 1)

        outputs = tf.keras.layers.Conv2D(self.num_classes, (3, 3), activation='sigmoid', padding='same')(decoder.output)
        self.model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
        adam = tf.keras.optimizers.Adam(lr=0.0001)
        self.model.compile(optimizer=adam,
                            loss=ModelsUtils.dice_coef_loss,
                            metrics=[ModelsUtils.dice_coef])
        self.model.summary()

        return self.model