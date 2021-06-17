import pickle
from Dataset.AIRS_Dataset.AIRS_Generator import AIRS_Generator
from Dataset.MassachusettsRoads_Dataset.MassachusettsGenerator import MassachusettsGenerator
from UNET.Models.AIRS.AIRS_UnetModel import AIRS_UnetModel
from UNET.Models.Massachusetts.MassachusettsUnetModel import MassachusettsUnetModel

class Trainer():

    def __init__(self, datasets):
        self.datasets = datasets

    def create_models(self):
        models = {}
        for dataset_key in self.datasets:
            if dataset_key == "airs":
                models[dataset_key] = AIRS_UnetModel(self.datasets[dataset_key].unet_params.patch_size,
                                           self.datasets[dataset_key].unet_params.patch_size,
                                           self.datasets[dataset_key].unet_params.image_channels)
            if dataset_key == "massachusetts":
                shape_size = (self.datasets[dataset_key].unet_params.patch_size,
                              self.datasets[dataset_key].unet_params.patch_size,
                              self.datasets[dataset_key].unet_params.image_channels)
                models[dataset_key] = MassachusettsUnetModel(1, shape_size)
        return models

    def create_generators(self):
        generators = {}
        for dataset_key in self.datasets:
            if dataset_key == "airs":
                generators[dataset_key] = {"train": AIRS_Generator(self.datasets[dataset_key].X_train_filenames,
                                                                    self.datasets[dataset_key].y_train_filenames,
                                                                    self.datasets[dataset_key].unet_params.batch_size),
                                           "validation": AIRS_Generator(self.datasets[dataset_key].X_val_filenames,
                                                                         self.datasets[dataset_key].y_val_filenames,
                                                                         self.datasets[dataset_key].unet_params.batch_size)}
            if dataset_key == "massachusetts":
                generators[dataset_key] = {"train": MassachusettsGenerator(self.datasets[dataset_key].X_train_filenames,
                                                                           self.datasets[dataset_key].y_train_filenames,
                                                                           self.datasets[dataset_key].unet_params.batch_size),
                                           "validation": MassachusettsGenerator(self.datasets[dataset_key].X_val_filenames,
                                                                                self.datasets[dataset_key].y_val_filenames,
                                                                                self.datasets[dataset_key].unet_params.batch_size)}

        return generators

    def start_learn(self):
        self.models = self.create_models()
        self.generators = self.create_generators()

        for model_key, generators_key, dataset_key in zip(self.models, self.generators, self.datasets):

            print("self.datasets[dataset_key].X_train_filenames.shape[0] ", self.datasets[dataset_key].X_train_filenames.shape[0])
            print("self.datasets[dataset_key].X_val_filenames.shape[0] ", self.datasets[dataset_key].X_val_filenames.shape[0] )
            print("generator=generators train ", self.generators[generators_key]["train"])
            print("generator=generators val ", self.generators[generators_key]["validation"])

            if model_key == "airs":
                self.airs_history = self.models[model_key].model.fit_generator(
                                                        generator=self.generators[generators_key]["train"],
                                                        steps_per_epoch=int(self.datasets[dataset_key].X_train_filenames.shape[0] // self.datasets[
                                                            dataset_key].unet_params.batch_size),
                                                        epochs=10,
                                                        validation_data=self.generators[generators_key]["validation"],
                                                        validation_steps=int(self.datasets[dataset_key].X_val_filenames.shape[0] // self.datasets[
                                                            dataset_key].unet_params.batch_size))
                self.save_model(self.models[model_key].model, 'AIRS_buildings.hdf5')
                self.save_history(self.massachusetts_history, "MassachusettsRoadsHistory")

            if model_key == "massachusetts":
                self.massachusetts_history = self.models[model_key].model.fit_generator(
                    generator=self.generators[generators_key]["train"],
                    steps_per_epoch=int(self.datasets[dataset_key].X_train_filenames.shape[0] // self.datasets[
                        dataset_key].unet_params.batch_size),
                    epochs=12,
                    # callbacks=callbacks,
                    validation_data=self.generators[generators_key]["validation"],
                    validation_steps=int(self.datasets[dataset_key].X_train_filenames.shape[0] // self.datasets[
                        dataset_key].unet_params.batch_size),
                    class_weight=None,
                    max_queue_size=10,
                    workers=1
                    )
                self.save_model(self.models[model_key].model, 'MassachusettsRoads.hdf5')
                self.save_history(self.massachusetts_history, "MassachusettsRoadsHistory")

    def save_model(self, model, filename):
        model.save(filename)

    def save_history(self, history, filename):
        with open(f'/{filename}', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)




