import tensorflow as tf

from UNET.Models.AIRS.AIRS_Decoder import AIRS_Decoder
from UNET.Models.AIRS.AIRS_Encoder import AIRS_Encoder
class AIRS_UnetModel():

    def __init__(self, WIDTH = 1024, HEIGHT= 1024, CHANNELS = 1):
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT
        self.CHANNELS = CHANNELS
        self.build_model()

    def build_model(self):

        # Build the model
        inputs = tf.keras.layers.Input((self.HEIGHT, self.WIDTH, self.CHANNELS))
        s = inputs

        encoder = AIRS_Encoder(s)
        decoder = AIRS_Decoder(encoder.skip_connections)

        outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(decoder.output)

        self.model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.model.summary()
        return self.model