from typing import List, Optional
import torch
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import Subset, random_split
import numpy as np

class MNISTDataset():
    def __init__(
            self,
            dataset_number_per_teacher: int,
            teacher_number: int,
            batch_size: int,
        ) -> None:
        super().__init__()
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.original_dataset = MNIST(root = './data', transform = transform, download=True)
        self.split_datasets: List[Subset[MNIST]] = []

        self.batch_size = batch_size
        self.dataset_number_per_teacher = dataset_number_per_teacher
        self.teacher_number = teacher_number

        self.split(dataset_number_per_teacher, teacher_number, batch_size)
        # TODO - device setting
        self.device = "cuda:0"

    def split(self, dataset_number_per_teacher, teacher_number, batch_size):
        original_data_len = len(self.original_dataset)
        dataset_number = dataset_number_per_teacher * teacher_number
        
        split_data_len = original_data_len // dataset_number
        print(original_data_len)
        self.dataset_len = split_data_len // batch_size + 1
        # print(original_data_len, split_data_len, self.dataset_len)
        self.split_datasets = random_split(
                self.original_dataset,
                [split_data_len for _ in range(dataset_number)]
            )

    def preprocess_y(self, data):
        ys = np.array([y for _, y in data])
        ys = torch.LongTensor(ys).view(-1, 1)

        batch_size = ys.shape[0]
        y_onehot = torch.FloatTensor(batch_size, 10)
        y_onehot.zero_()
        y_onehot.scatter_(1, ys, 1)

        return y_onehot


    def get_data(self, teacher_idx, teacher_data_idx, batch_idx):
        # print(len(self.split_datasets), teacher_idx * self.dataset_number_per_teacher + teacher_data_idx)
        split_dataset = self.split_datasets[teacher_idx * self.dataset_number_per_teacher + teacher_data_idx]
        start = self.batch_size * batch_idx
        end = min(start + self.batch_size, len(split_dataset))
        data = [split_dataset[i] for i in range(start, end)]
        xs = torch.stack([x for x, _ in data], dim=0).to(self.device)
        ys = self.preprocess_y(data).to(self.device)

        return xs, ys

    def get_random_label(self, batch_size):
        indices = np.random.choice(len(self.original_dataset), batch_size)
        
        data = [self.original_dataset[i] for i in indices]
        ys = self.preprocess_y(data).to(self.device)

        return ys


    def len(self):
        if len(self.split_datasets) == 0:
            return 0
        return len(self.split_datasets[0]) // self.batch_size + 1

    def __repr__(self) -> str:
        return f'<MNISTDataset teacher: {self.teacher_number}, dataset_number_per_teacher: {self.dataset_number_per_teacher}>'
