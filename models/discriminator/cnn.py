import torch
import torch.nn as nn
from .base import Discriminator

class MNISTDiscriminator(Discriminator):
    def __init__(self):
        super(MNISTDiscriminator, self).__init__()

        self.layer1 = nn.Sequential(
            # Input is B x 11 x 28 x 28
            nn.Conv2d(11, 64, (4, 4), (2, 2), (1, 1), bias=True),
            nn.LeakyReLU(0.2, True),
        )
        self.layer2 = nn.Sequential(
            # State size. B x 74 x 14 x 14
            nn.Conv2d(74, 128, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
        )
        self.layer3 = nn.Sequential(
            # State size. B x (128 * 7 * 7 + 10)
            nn.Linear(128 * 7 * 7 + 10, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, True),
        )

        self.layer4 = nn.Sequential(
            # State size. B x 266
            nn.Linear(266, 1),
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

    def _forward_with_no_condition(self, x) -> torch.Tensor:
        # TODO - not in paper, so we need to decide what to make model
        raise NotImplementedError("forward with no condition must be implemented")

    def _forward_with_condition(self, x, y) -> torch.Tensor:
        # Input is x: B x 1 x 64 x 64, y: B x 10

        y_image = torch.reshape(y, [y.shape[0], y.shape[1], 1, 1])
        layer1_input = self._conv_cond_concat(x, y_image)
        layer1_output = self.layer1(layer1_input)
        layer2_input = self._conv_cond_concat(layer1_output, y_image)
        layer2_output = self.layer2(layer2_input)
        layer3_input = torch.concat(
                [
                    torch.reshape(layer2_output, [layer2_output.shape[0], -1]),
                    y
                ],
                dim = 1
            )
        layer3_output = self.layer3(layer3_input)
        layer4_input = torch.concat([layer3_output, y], dim = 1)
        layer4_output = self.layer4(layer4_input)
        return layer4_output
