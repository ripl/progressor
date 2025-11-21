import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class RandomApplyTransform:
    ''' Apply a list of transforms in random order with 50% probability each.
    '''
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        transforms = list(self.transforms)
        random.shuffle(transforms)
        for t in transforms:
            if random.random() < 0.5:
                img = t(img)
        return img


class RandomShiftsAug(nn.Module):
    ''' Randomly shift the image by a small amount.
    '''
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        x = x.unsqueeze(0)
        n, _, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        out = F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)
        return out.squeeze(0)


class GaussianNoise(nn.Module):
    ''' Add Gaussian noise to the input.
    '''
    def __init__(self, mean=0., std=1.0):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        if isinstance(self.std, tuple):
            std = torch.empty(1).uniform_(self.std[0], self.std[1]).item()
        else:
            std = self.std
        noise = torch.randn_like(x) * std + self.mean
        return x + noise
    