import matplotlib.colors as colors
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMessageBox


class Utils:

    @staticmethod
    def hex2rgb(value):
        rgb = colors.hex2color(value)
        return tuple([int(255 * x) for x in rgb])

    @staticmethod
    def rgb2hex(value):
        return '#%02X%02X%02X' % value

    @staticmethod
    def show_message_box(title: str, message: str, icon: QMessageBox.Icon):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setText(message)
        msg.setIcon(icon)
        msg.setWindowTitle(title)
        msg.setStandardButtons(QMessageBox.Cancel)
        msg.exec_()

    @staticmethod
    def arrimage2QPixmap(array):
        height, width, channel = array.shape
        bytesPerLine = 3 * width
        qImg = QImage(array.data, width, height, bytesPerLine, QImage.Format_RGB888)
        return QPixmap(qImg)

    @staticmethod
    def numToFixed(numObj, digits=0):
        return f"{numObj:.{digits}f}"
