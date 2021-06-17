from matplotlib import pyplot as plt
import numpy as np

class TrainStatistics():

    def __init__(self, model, history, data_loader):
        self.model = model
        self.history = history
        self.data_loader = data_loader

    def plot_acc_and_loss(self):
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']
        epochs = range(1, len(loss) + 1)
        plt.plot(epochs, loss, 'y', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        acc = self.history.history['acc']
        # acc = history.history['accuracy']
        val_acc = self.history.history['val_acc']
        # val_acc = history.history['val_accuracy']

        plt.plot(epochs, acc, 'y', label='Training acc')
        plt.plot(epochs, val_acc, 'r', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

    def calculate_iou(self):
        y_pred = self.model.predict(self.model.X_)
        y_pred_thresholded = y_pred > 0.5

        intersection = np.logical_and(self.data_loader.y_test, y_pred_thresholded)
        union = np.logical_or(self.y_test, y_pred_thresholded)
        iou_score = np.sum(intersection) / np.sum(union)
        print("IoU socre is: ", iou_score)

    def evaluate_model(self):
        # evaluate model
        _, self.acc = self.unet.model.evaluate(self.X_test, self.y_test)
        print("Accuracy = ", (self.acc * 100.0), "%")