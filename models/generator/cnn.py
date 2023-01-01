import torch
import torch.nn as nn
from .base import Generator

class MNISTGenerator(Generator):
    def __init__(self, latent_dim):
        super(MNISTGenerator, self).__init__()
        self.gfc_dim = 1024
        self.gf_dim = 64
        self.c_dim = 1
        self.label_dim = 10

        self.layer1 = nn.Sequential(
            # state size. B x (z + 10)
            nn.Linear(latent_dim + self.label_dim, self.gfc_dim),
            nn.BatchNorm1d(self.gfc_dim),
        )
        self.layer2 = nn.Sequential(
            # state size. B x 1024
            nn.Linear(self.gfc_dim + self.label_dim, self.gf_dim * 2 * 7 * 7),
            nn.BatchNorm1d(self.gf_dim * 2 * 7 * 7),
            nn.ReLU(True),
        )
        self.layer3 = nn.Sequential(
            # state size. (128 + 10) x 7 x 7
            nn.ConvTranspose2d(self.gf_dim * 2 + self.label_dim, self.gf_dim * 2, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(self.gf_dim * 2),
            nn.ReLU(True),
        )
        self.layer4 = nn.Sequential(
            # state size. (128 + 10) x 14 x 14
            nn.ConvTranspose2d(self.gf_dim * 2 + self.label_dim, self.c_dim, (4, 4), (2, 2), (1, 1), bias=False),
            nn.Sigmoid()
        )


    def _conv_cond_concat(self, x, y):
        """
        concat at channel dimension
        ex)
            x: B x 1 x 64 x 64, y : B x 11 x 1 x 1
            out: B x 11 x 64 x 64
        """
        return torch.concat(
                [
                    x,
                    y * torch.ones([x.shape[0], y.shape[1], x.shape[2], x.shape[3]], device=y.device)
                ],
                dim = 1
            )

    def _forward_with_no_condition(self, z) -> torch.Tensor:
        # TODO - not in paper, so we need to decide what to make model
        raise NotImplementedError("forward with no condition must be implemented")
 
    def _forward_with_condition(self, z, y) -> torch.Tensor:
        y_image = torch.reshape(y, [y.shape[0], y.shape[1], 1, 1])
        layer1_input = torch.concat([z, y], dim = 1)
        layer1_output = self.layer1(layer1_input)
        layer2_input = torch.concat([layer1_output, y], dim = 1)
        layer2_output = self.layer2(layer2_input)

        layer3_input = self._conv_cond_concat(
            torch.reshape(
                layer2_output,
                [y.shape[0], self.gf_dim * 2, 7, 7]
            ),
            y_image
        )
        # layer3_input = self._conv_cond_concat(layer2_output, y_image)
        layer3_output = self.layer3(layer3_input)
        layer4_input = self._conv_cond_concat(layer3_output, y_image)
        layer4_output = self.layer4(layer4_input)
        return layer4_output

