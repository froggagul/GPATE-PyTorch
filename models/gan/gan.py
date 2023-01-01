import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.generator import MNISTGenerator
from models.discriminator import MNISTDiscriminator
from models.aggregator import AggregatorFactory, Aggregator
from typing import Optional
from logger import EpochDataStore

def adversarial_loss(y_hat, y):
    return F.binary_cross_entropy(y_hat, y)


class GAN(nn.Module):
    def __init__(self) -> None:
        super(GAN, self).__init__()
        self.latent_dim = 8

        self.teacher_number = 100
        self.split_data_number = 4000
        assert self.split_data_number % self.teacher_number == 0, "split_data_number must be multiples of teacher_number"
        self.shared_data_number_per_teacher = self.split_data_number // self.teacher_number

        self.lr = 0.01
        self.b1 = 0.5
        self.b2 = 0.999
        self.max_eps = 10

        self.loss = adversarial_loss

        # TODO - generator, discriminator step per single epoch
        self.generator_step_in_single_epoch = 1
        self.discriminator_step_in_single_epoch = 1

        self.generator = MNISTGenerator(latent_dim = self.latent_dim)
        self.discriminators = nn.ModuleList([MNISTDiscriminator() for _ in range(self.teacher_number)])
    
        self.configure_optimizers()

        self.store: Optional[EpochDataStore] = None

        self.aggregator: Aggregator = AggregatorFactory.create('private_aggregator')

    def configure_optimizers(self):
        lr, b1, b2 = self.lr, self.b1, self.b2

        optimizer_discriminators: list[optim.Adam] = []
        for discriminator in self.discriminators:
            optimizer = torch.optim.Adam(discriminator.parameters(), lr = lr, betas = (b1, b2))
            optimizer_discriminators.append(optimizer)

        self.optimizer_discriminators = optimizer_discriminators

        self.optimizer_generator = torch.optim.Adam(
            self.generator.parameters(), lr = lr, betas = (b1, b2)
            )

    def reset_latent_vector(self, batch_size, device):
        z = torch.randn(batch_size, self.latent_dim, device=device)
        self._cached_z = z

    def get_latent_vector(self, batch_size):
        assert self._cached_z is not None, "need to reset z first, call reset_latent_vector(x)"

        return self._cached_z[:batch_size]

    def train_teacher(self, batch, teacher_idx):
        optimizer = self.optimizer_discriminators[teacher_idx]
        teacher_model = self.discriminators[teacher_idx]
        optimizer.zero_grad()
        z, (x, y) = batch

        with torch.no_grad():
            fake_x = self.generator(z, y)
            # student generator must not be updated while teacher is training
        fake_x.requires_grad = True

        fake_y_hat = teacher_model(fake_x, y)
        fake_y = torch.zeros(x.shape[0], 1, device=x.device)

        valid_y_hat = teacher_model(x, y)
        valid_y = torch.ones(x.shape[0], 1, device=x.device)

        fake_loss = self.loss(fake_y_hat, fake_y)
        valid_loss = self.loss(valid_y_hat, valid_y)

        loss = fake_loss + valid_loss
        loss.backward()
        optimizer.step()

        return torch.reshape(fake_x.grad, [x.shape[0], -1]).clone()

    def train_student(self, total_grads, batch, epoch):
        optimizer = self.optimizer_generator
        optimizer.zero_grad()
        z, y = batch

        perturbation = self._aggregate_grads(total_grads, epoch)
        fake_x = self.generator(z, y)
        perturbation = torch.reshape(perturbation, fake_x.shape)

        fake_x_update = (fake_x + perturbation).detach()

        loss = F.mse_loss(fake_x, fake_x_update)
        loss.backward()
        optimizer.step()

    def _aggregate_grads(self, total_grads, epoch):
        perturbation = self.aggregator.aggregate(total_grads, epoch)
        self.store.log('orders', self.aggregator.orders)
        self.store.log('rdp_counter', self.aggregator.rdp_counter)
        self.store.log('dp_delta', self.aggregator.dp_delta)

        return perturbation

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def connect_store(self, store: EpochDataStore):
        self.store = store
