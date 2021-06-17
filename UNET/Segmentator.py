import cv2
import numpy as np
from keras.utils import normalize
from patchify import patchify, unpatchify
from tqdm import tqdm

from Dataset.DatasetUtils import DatasetUtils
from UI.Utils import Utils
from UNET.Models.AIRS.AIRS_UnetModel import AIRS_UnetModel
from UNET.Models.ISPRS.ISPRS_UnetModel import ISPRS_UnetModel

from UNET.Models.Massachusetts.MassachusettsUnetModel import MassachusettsUnetModel


class Segmentator():

    def __init__(self, path, segmented_items):
        self.path = path
        self.segmented_items = segmented_items
        self.segmentation_models = self.load_segmentation_models()

    def load_segmentation_models(self):
        segmentation_models = {}
        for item_key in self.segmented_items:
            if item_key == "buildings":
                segmentation_model = AIRS_UnetModel(1024, 1024, 1)
                segmentation_model.model.load_weights(
                    '../UNET/Models/AIRS/AIRS_buildings.hdf5')  # Trained for 50 epochs and then additional 100
                segmentation_models[item_key] = segmentation_model.model

            if item_key == "roads":
                segmentation_model = MassachusettsUnetModel(1, (1024, 1024, 3))
                segmentation_model.model.load_weights(
                    '../UNET/Models/Massachusetts/Massachusetts_roads.hdf5')  # Trained for 50 epochs and then additional 100
                segmentation_models[item_key] = segmentation_model.model

            if item_key == "cars":
                segmentation_model = ISPRS_UnetModel(1, (512, 512, 1))
                segmentation_model.model.load_weights(
                    '../UNET/Models/ISPRS/ISPRS_cars.hdf5')  # Trained for 50 epochs and then additional 100
                segmentation_models[item_key] = segmentation_model.model
        return segmentation_models

    def run_segmentation(self):
        segmentation_results = dict.fromkeys([model_key for model_key in self.segmentation_models])

        for model_key in self.segmentation_models:
            if model_key == "buildings":
                segmentation_results[model_key] = self.run_build_segmentation(self.segmentation_models[model_key])

            if model_key == "roads":
                segmentation_results[model_key] = self.run_roads_segmentation(self.segmentation_models[model_key])

            if model_key == "cars":
                segmentation_results[model_key] = self.run_cars_segmentation(self.segmentation_models[model_key])

        segmented_image = self.combine_segmentation_results(segmentation_results)
        return segmented_image


    def combine_segmentation_results(self, segmentation_results):

        coloring_images = []
        for seg_result_key in segmentation_results:
            print(segmentation_results[seg_result_key][0])
            if seg_result_key == "buildings" or seg_result_key == "cars":
                segmentation_results[seg_result_key] = cv2.cvtColor(cv2.resize(segmentation_results[seg_result_key], (1024, 1024)), cv2.COLOR_GRAY2RGB)
                segmentation_results[seg_result_key] = self.binaryImage_buildings(segmentation_results[seg_result_key])
            print(segmentation_results[seg_result_key].shape)
            print(np.unique(segmentation_results[seg_result_key]))
            coloring_images.append(self.coloring_image(segmentation_results[seg_result_key], self.segmented_items[seg_result_key].text()))
        segmented_image = self.combine_colorized_images(coloring_images)
        return segmented_image

    def coloring_image(self, img, color):
        color_rgb = Utils.hex2rgb(color)
        img[(img[:, :, 0] > 0) & (img[:, :, 1] > 0) & (img[:, :, 2] > 0)] = [color_rgb[0], color_rgb[1], color_rgb[2]]
        color_image = img
        return color_image

    def combine_colorized_images(self, images):
        print(images[0].shape)
        h, w, c = images[0].shape
        segmented_image = np.zeros((h, w, 3), np.uint8)
        for image in images:
            image = image.astype(np.uint8)
            segmented_image += image
        return segmented_image


    def run_build_segmentation(self, model):
        large_image = cv2.imread(self.path, 0)
        img = DatasetUtils.resize_2d_img(large_image, 1024)
        patches = patchify(img, (1024, 1024), step=1024)

        predicted_patches = []
        for i in range(patches.shape[0]):
            for j in tqdm(range(patches.shape[1])):
                single_patch = patches[i, j, :, :]
                single_patch_norm = np.expand_dims(normalize(np.array(single_patch), axis=1), 2)
                single_patch_input = np.expand_dims(single_patch_norm, 0)
                single_patch_prediction = (model.predict(single_patch_input)[0, :, :, 0] > 0.5).astype(np.uint8)
                predicted_patches.append(single_patch_prediction)

        predicted_patches = np.array(predicted_patches)
        predicted_patches_reshaped = np.reshape(predicted_patches,
                                                (patches.shape[0], patches.shape[1], 1024, 1024))

        reconstructed_image = unpatchify(predicted_patches_reshaped, img.shape)
        return reconstructed_image

    def run_roads_segmentation(self, model):
        img = cv2.imread(self.path)
        x_batch = []
        img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_AREA)
        cv2.imwrite('D:/src_seg_img_1005.png', img)
        cv2.imwrite('D:/src_seg_img_1005.tif', img)
        x_batch += [img]
        x_batch = np.array(x_batch) / 255.
        predicted_image = model.predict(x_batch, verbose=1)
        predicted_image_bin = self.binaryImage_roads(predicted_image)
        return predicted_image_bin

    def run_cars_segmentation(self, model):
        large_image = cv2.imread(self.path, 0)
        img = DatasetUtils.resize_2d_img(large_image, 512)
        patches = patchify(img, (512, 512), step=512)

        predicted_patches = []
        for i in range(patches.shape[0]):
            for j in tqdm(range(patches.shape[1])):
                single_patch = patches[i, j, :, :]
                single_patch_norm = np.expand_dims(single_patch, 2)
                single_patch_input = np.expand_dims(single_patch_norm, 0)
                single_patch_prediction = (model.predict(single_patch_input)[0, :, :, 0] > 0.5).astype(np.uint8)
                predicted_patches.append(single_patch_prediction)

        predicted_patches = np.array(predicted_patches)
        predicted_patches_reshaped = np.reshape(predicted_patches,
                                                (patches.shape[0], patches.shape[1], 512, 512))

        reconstructed_image = unpatchify(predicted_patches_reshaped, img.shape)
        return reconstructed_image

    def binaryImage_roads(self, image):
        x = image.shape[1]
        y = image.shape[2]
        imgs = np.zeros((x, y, 3))
        for k in range(x):
            for n in range(y):
                if image[0, k, n] > 0.5:
                    imgs[k, n, 0] = 255
                    imgs[k, n, 1] = 255
                    imgs[k, n, 2] = 255
        return imgs

    def binaryImage_buildings(self, image):
        x = image.shape[0]
        y = image.shape[1]
        imgs = np.zeros((x, y, 3))
        for k in range(x):
            for n in range(y):
                if image[k, n, 0] > 0.5:
                    imgs[k, n, 0] = 255
                    imgs[k, n, 1] = 255
                    imgs[k, n, 2] = 255
        return imgs