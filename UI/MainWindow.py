import os
import sys
import re
import cv2
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QColorDialog, QMessageBox
from Dataset.AIRS_Dataset.AIRS_Dataset import AIRS_Dataset
from Dataset.MassachusettsRoads_Dataset.MassachusettsDataset import MassachusettsDataset
from UI.ui import Ui_MainWindow
from UI.Utils import Utils
from UNET.Segmentator import Segmentator
from UNET.Trainer import Trainer
import time
import numpy as np


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # LearningTab
        self.loading_datasets_count = 0
        self.block_ui_items()
        self.set_buttons_connects()

        # SegmentationTab
        self.seg_properties_ui = {
            "cars": {
                "checkBox": self.ui.checkBox_cars,
                "color_button": self.ui.color_button_4,
                "rgb_spinBox": (self.ui.R_spinBox_cars, self.ui.G_spinBox_cars, self.ui.B_spinBox_cars),
                "hex_lineEdit": self.ui.HEX_lineEdit_cars
            },
            "roads": {
                "checkBox": self.ui.checkBox_road,
                "color_button": self.ui.color_button_2,
                "rgb_spinBox": (self.ui.R_spinBox_road, self.ui.G_spinBox_road, self.ui.B_spinBox_road),
                "hex_lineEdit": self.ui.HEX_lineEdit_road
            },
            "buildings": {
                "checkBox": self.ui.checkBox_buildings,
                "color_button": self.ui.color_button_3,
                "rgb_spinBox": (self.ui.R_spinBox_buildings, self.ui.G_spinBox_buildings, self.ui.B_spinBox_buildings),
                "hex_lineEdit": self.ui.HEX_lineEdit_buildings
            },
            # "foressetTristatets": {
            #     "checkBox": self.ui.checkBox_forests,
            #     "color_button": self.ui.color_button_4,
            #     "rgb_spinBox": (self.ui.R_spinBox_forests, self.ui.G_spinBox_forests, self.ui.B_spinBox_forests),
            #     "hex_lineEdit": self.ui.HEX_lineEdit_forests
            # },
        }
        self.set_default_color_values()
        self.set_seg_properties_connects()

    def set_seg_properties_connects(self):
        # ColorPickers
        self.ui.color_button_4.clicked.connect(lambda: self.on_color_picker_clicked(self.seg_properties_ui["cars"]))
        self.ui.color_button_2.clicked.connect(lambda: self.on_color_picker_clicked(self.seg_properties_ui["roads"]))
        self.ui.color_button_3.clicked.connect(lambda: self.on_color_picker_clicked(self.seg_properties_ui["buildings"]))
        # self.ui.color_button_4.clicked.connect(lambda: self.on_color_picker_clicked(self.seg_properties_ui["forests"]))

        # SpinBoxes
        self.ui.R_spinBox_cars.valueChanged.connect(self.on_spinBox_value_changed)
        self.ui.R_spinBox_road.valueChanged.connect(self.on_spinBox_value_changed)
        self.ui.R_spinBox_buildings.valueChanged.connect(self.on_spinBox_value_changed)
        # self.ui.R_spinBox_forests.valueChanged.connect(self.on_spinBox_value_changed)

        self.ui.G_spinBox_cars.valueChanged.connect(self.on_spinBox_value_changed)
        self.ui.G_spinBox_road.valueChanged.connect(self.on_spinBox_value_changed)
        self.ui.G_spinBox_buildings.valueChanged.connect(self.on_spinBox_value_changed)
        # self.ui.G_spinBox_forests.valueChanged.connect(self.on_spinBox_value_changed)

        self.ui.B_spinBox_cars.valueChanged.connect(self.on_spinBox_value_changed)
        self.ui.B_spinBox_road.valueChanged.connect(self.on_spinBox_value_changed)
        self.ui.B_spinBox_buildings.valueChanged.connect(self.on_spinBox_value_changed)
        # self.ui.B_spinBox_forests.valueChanged.connect(self.on_spinBox_value_changed)

        #HEX_LineEdits

        self.ui.HEX_lineEdit_cars.textEdited.connect(self.on_lineEdit_value_changed)
        self.ui.HEX_lineEdit_road.textEdited.connect(self.on_lineEdit_value_changed)
        self.ui.HEX_lineEdit_buildings.textEdited.connect(self.on_lineEdit_value_changed)
        # self.ui.HEX_lineEdit_forests.textEdited.connect(self.on_lineEdit_value_changed)

        #SegmentationImageButton
        self.ui.browse_image_button.clicked.connect(self.on_browse_seg_image_button_clicked)
        self.ui.segmentation_button.clicked.connect(self.on_segmentation_button_clicked_)
        self.ui.download_png_button.clicked.connect(self.download_png)
        self.ui.download_tif_button.clicked.connect(self.download_tif)

    def download_png(self):
        name = QFileDialog.getSaveFileName(self, 'Save File', filter="(*.png)")
        cv2.imwrite(name[0], self.segmented_image)

    def download_tif(self):
        name = QFileDialog.getSaveFileName(self, 'Save File', filter="(*.tif)")
        cv2.imwrite(name[0], self.segmented_image)

    def on_segmentation_button_clicked_(self):

        # properties is valid
        dict_items = self.seg_properties_ui.items()
        # checkBoxes
        check_boxes = [item[1]['checkBox'] for item in dict_items]
        is_no_segmented_objects_selected = False
        for cb in check_boxes:
            if cb.isChecked():
                is_no_segmented_objects_selected = True
                break
        if not is_no_segmented_objects_selected:
            Utils.show_message_box("Error", "Select the objects to segment", QMessageBox.Critical)
            return

        # Segmentation
        segmented_items = {}
        if self.ui.checkBox_buildings.isChecked():
            segmented_items["buildings"] = self.ui.HEX_lineEdit_buildings
        if self.ui.checkBox_road.isChecked():
            segmented_items["roads"] = self.ui.HEX_lineEdit_road
        if self.ui.checkBox_cars.isChecked():
            segmented_items["cars"] = self.ui.HEX_lineEdit_cars

        segmentator = Segmentator(self.segmentation_image_filename, segmented_items)

        start_time = time.time()
        self.segmented_image = segmentator.run_segmentation()
        end_time = time.time()
        segmented_time = Utils.numToFixed(end_time - start_time, 2)

        self.ui.time_value.setText('Seconds: ' + segmented_time)

        self.ui.label_32.setPixmap(Utils.arrimage2QPixmap(self.segmented_image))

    def on_browse_seg_image_button_clicked(self):
        self.segmentation_image_filename, _filter = QFileDialog.getOpenFileName(self, 'Open file','', "Images (*.png *.jpg *.tif)")
        img = cv2.imread(self.segmentation_image_filename)
        self.ui.segmentation_image_label.setPixmap(Utils.arrimage2QPixmap(img))
        self.ui.label_31.setPixmap(Utils.arrimage2QPixmap(img))

    def set_default_color_values(self):
        defaultColors = ["#F9C22E", "#F15946", "#E01A4F", "#0C090D"]
        defaultColors_rgb = [Utils.hex2rgb(color) for color in defaultColors]
        dict_items = self.seg_properties_ui.items()
        for i, item in enumerate(dict_items):
            item[1]["hex_lineEdit"].setText(defaultColors[i])
            item[1]["color_button"].setStyleSheet(f"background-color: rgb({defaultColors_rgb[i][0]}, {defaultColors_rgb[i][1]}, {defaultColors_rgb[i][2]})")
            item[1]["rgb_spinBox"][0].setValue(defaultColors_rgb[i][0])
            item[1]["rgb_spinBox"][1].setValue(defaultColors_rgb[i][1])
            item[1]["rgb_spinBox"][2].setValue(defaultColors_rgb[i][2])


    def on_spinBox_value_changed(self, properties):
        print("SPINBOX")
        dict_items = self.seg_properties_ui.items()
        for i, item in enumerate(dict_items):
            r, g, b = item[1]["rgb_spinBox"][0].value(), item[1]["rgb_spinBox"][1].value(), item[1]["rgb_spinBox"][2].value()
            item[1]["hex_lineEdit"].setText(Utils.rgb2hex((r, g, b)))
            item[1]["color_button"].setStyleSheet(f"background-color: rgb({r}, {g}, {b})")

    def on_lineEdit_value_changed(self, properties):
        print("LINEEDIT")
        dict_items = self.seg_properties_ui.items()
        for i, item in enumerate(dict_items):
            hex = item[1]["hex_lineEdit"].text()
            if re.search(r'^#(?:[0-9a-fA-F]{3}){1,2}$', hex):
                rgb = Utils.hex2rgb(hex)
                item[1]["color_button"].setStyleSheet(f"background-color: rgb({rgb[0]}, {rgb[1]}, {rgb[2]})")
                item[1]["rgb_spinBox"][0].setValue(rgb[0])
                item[1]["rgb_spinBox"][1].setValue(rgb[1])
                item[1]["rgb_spinBox"][2].setValue(rgb[2])

    def on_color_picker_clicked(self, properties):
        color_dialog = QColorDialog()
        color = color_dialog.getColor()
        rgb_tuple = (color.red(), color.green(), color.blue())

        properties["color_button"].setStyleSheet(f"background-color: rgb({rgb_tuple[0]}, {rgb_tuple[1]}, {rgb_tuple[2]})")
        properties["rgb_spinBox"][0].setValue(rgb_tuple[0])
        properties["rgb_spinBox"][1].setValue(rgb_tuple[1])
        properties["rgb_spinBox"][2].setValue(rgb_tuple[2])
        properties["hex_lineEdit"].setText(Utils.rgb2hex(rgb_tuple))

# LearningTab
    def block_ui_items(self):
        pass
        # self.ui.tab_widget.setTabEnabled(1, False)
        # self.ui.tab_widget.setTabEnabled(2, False)
        # self.ui.Learning.setTabEnabled(1, False)
        # self.ui.start_learning_button.setEnabled(False)


    def set_buttons_connects(self):
        self.ui.start_learning_button.clicked.connect(self.learning)

    def load_data(self, path, layout):
        filenames, _filter = QFileDialog.getOpenFileNames(self, 'Open file', '',"Images (*.png *.jpg)")
        if not filenames:
            return
        for i in range(layout.count()):
            layout.itemAt(i).widget().setPixmap(QPixmap(filenames[i]))

        self.loading_datasets_count += 1

    def learning(self):

        datasets = {}
        if self.ui.MassachusettsDatasetCheckbox.isChecked():
            self.massachusetts_dataset = MassachusettsDataset()
            datasets["massachusetts"] = self.massachusetts_dataset

        if self.ui.AirsDatasetCheckbox.isChecked():
            self.airs_dataset = AIRS_Dataset()
            datasets["airs"] = self.airs_dataset

        if not datasets:
            Utils.show_message_box("Error", "Select datasets for training", QMessageBox.Critical)
            return

        self.trainer = Trainer(datasets)

        self.trainer.start_learn()

if __name__ == '__main__':
    app = QApplication([])
    application = MainWindow()
    application.show()
    sys.exit(app.exec())