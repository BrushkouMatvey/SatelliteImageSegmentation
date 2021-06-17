from keras import backend as K
from keras.backend import binary_crossentropy

class ModelsUtils():

    @staticmethod
    def dice_coef(y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)

    @staticmethod
    def dice_coef_loss(y_true, y_pred):
        return 1 - ModelsUtils.dice_coef(y_true, y_pred)

    smooth = 1e-12

    @staticmethod
    def jaccard_coef(y_true, y_pred):
        intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
        sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])

        jac = (intersection + 1e-12) / (sum_ - intersection + 1e-12)

        return K.mean(jac)

    @staticmethod
    def jaccard_coef_int(y_true, y_pred):
        y_pred_pos = K.round(K.clip(y_pred, 0, 1))

        intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
        sum_ = K.sum(y_true + y_pred_pos, axis=[0, -1, -2])

        jac = (intersection + 1e-12) / (sum_ - intersection + 1e-12)

        return K.mean(jac)

    @staticmethod
    def jaccard_coef_loss(y_true, y_pred):
        return -K.log(ModelsUtils.jaccard_coef(y_true, y_pred)) + binary_crossentropy(y_pred, y_true)