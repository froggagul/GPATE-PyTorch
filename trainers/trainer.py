import torch
import torch.nn as nn
import torch.optim as optim
from models.gan import GAN
from dataset import MNISTDataset
from logger import EpochDataStore
from pytorch_lightning.loggers import Logger
from typing import Optional, List, Type
from callbacks import Callback, DPCallback

class Trainer():
    def __init__(self):

        self.model: GAN = GAN()
        self.batch_size = 32
        self.dataset: MNISTDataset = MNISTDataset(
                dataset_number_per_teacher=10,
                teacher_number=self.model.teacher_number,
                batch_size = self.batch_size
            )
        self.callback_classes: List[Type[Callback]] = [
        ]
        self.callbacks: List[Callback] = []

        self.device = "cuda:0"
        self.current_epoch: int = -1
        self.logger: Optional[Logger] = None
        self.store: Optional[EpochDataStore] = None

        self._is_end_of_epoch = False

    def log(self, key, value):
        print(key, value)

    def call_end_of_loop(self):
        self._is_end_of_epoch = True

    def check_end_of_epoch(self):
        return self._is_end_of_epoch

    def _single_epoch(self):
        batch_size = self.batch_size
        self.model.reset_latent_vector(batch_size, self.device)
        # TODO - require batch idx for logging

        for batch_idx in range(self.dataset.len()):
            total_grads = []
            for teacher_data_idx in range(self.model.shared_data_number_per_teacher):
                for teacher_idx in range(self.model.teacher_number):
                    data = self.dataset.get_data(teacher_idx, teacher_data_idx, batch_idx)
                    z = self.model.get_latent_vector()
                    batch = (z, data)
                    grads = self.model.train_teacher(batch, teacher_idx)
                    total_grads.append(grads)

            y_label = self.dataset.get_random_label(batch_size)
            z = self.model.get_latent_vector()
            batch = (z, y_label)
            self.model.train_student(total_grads, batch, self.current_epoch)

    def _step_epoch(self, epoch):
        self.current_epoch = epoch
        
        if self.model.store:
            self.model.store.step_epoch()

    def _initialize_callbacks(self):
        for callback_class in self.callback_classes:
            self.callbacks.append(callback_class(self.store, self, self.logger))

    def _call_callbacks(self):
        for callback in self.callbacks:
            callback()

    def fit(self):
        # TODO - get param as dataset, model, logger, callback_classes
        self.model: GAN = GAN()
        self.batch_size = 32
        self.store = EpochDataStore()

        self.callback_classes = [DPCallback]
        self._initialize_callbacks()

        self.dataset: MNISTDataset = MNISTDataset(
                dataset_number_per_teacher=10,
                teacher_number=self.model.teacher_number,
                batch_size = self.batch_size
            )

        self.model = self.model.to(self.device)
        self.model.connect_store(self.store)

        self.model.train()
        epochs = 2
        self.current_epoch = 0
        for epoch in range(epochs):
            self._step_epoch(epoch)
            self._single_epoch()

            self._call_callbacks()

            if self.check_end_of_epoch():
                break
            
        return
