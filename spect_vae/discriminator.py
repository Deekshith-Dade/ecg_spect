from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectrogramPatchDiscriminator(nn.Module):
    def __init__(self, in_channels=1, base_channels=64, n_layers=3, use_spectral_norm=True):
        super().__init__()
        
        self.discriminators = nn.ModuleList([
            PatchDiscriminator(in_channels, base_channels, n_layers, use_spectral_norm)
            for _ in range(3)
        ])

        self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=False)
    
    def forward(self, x):
        outputs = []
        for disc in self.discriminators:
            outputs.append(disc(x))
            x = self.downsample(x)
        
        return outputs

class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels, base_channels=64, n_layers=3, use_spectral_norm=True):
        super().__init__()

        norm_layer = nn.BatchNorm2d
        if use_spectral_norm:
            norm_fn = lambda layer: nn.utils.spectral_norm(layer)
        else:
            norm_fn = lambda layer: layer
        
        layers = [
            norm_fn(nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            layers += [
                norm_fn(nn.Conv2d(
                    base_channels * nf_mult_prev,
                    base_channels * nf_mult,
                    kernel_size = 4,
                    stride = 2
                )),
                norm_layer(base_channels*nf_mult),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        layers += [
            norm_fn(nn.Conv2d(
                base_channels * nf_mult_prev,
                base_channels * nf_mult,
                kernel_size = 4,
                stride = 1,
                padding = 1
            )),
            norm_layer(base_channels * nf_mult),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        layers += [
            norm_fn(nn.Conv2d(base_channels * nf_mult, 1, kernel_size=4, stride=1, padding=1))
        ]

        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

class GANLoss(nn.Module):
    def __init__(self, loss_type="hinge"):
        super().__init__()
        self.loss_type = loss_type
        
        if loss_type == 'hinge':
            self.loss_fn = self.hinge_loss
        elif loss_type == 'non_saturating':
            self.loss_fn = self.non_saturating_loss
        elif loss_type == 'least_squares':
            self.loss_fn = self.least_squares_loss
        else:
            raise ValueError(f"Unknown Loss type: {loss_type}")
        
    def hinge_loss(self, pred, is_real):
        if is_real:
            loss = torch.mean(F.relu(1.0 - pred))
        else:
            loss = torch.mean(F.relu(1.0 + pred))
        return loss

    def non_saturating_loss(self, pred, is_real):
        if is_real:
            loss = F.softplus(-pred).mean()
        else:
            loss = F.softplus(pred).mean()
        return loss
    
    def least_squares_loss(self, pred, is_real):
        if is_real:
            loss = F.mse_loss(pred, torch.ones_like(pred))
        else:
            loss = F.mse_loss(pred, torch.zeros_like(pred))
        return loss
    
    def forward(self, pred, is_real):
        if isinstance(pred, list):
            loss = 0
            for p in pred:
                loss += self.loss_fn(p, is_real)
            return loss / len(pred)
        else:
            return self.loss_fn(pred, is_real)
        