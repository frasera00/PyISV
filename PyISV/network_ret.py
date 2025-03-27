import numpy as np
import torch
from torch import nn

class Autoencoder2D(nn.Module):

    def __init__(self, embed_dim, flat_dim, num_encoder_final_channels=128):
        super(Autoencoder2D, self).__init__()
        ## parameters of the network
        self.embed_dim = embed_dim
        self.flat_dim = flat_dim
        self.num_final_ch = num_encoder_final_channels

        ## encoder architecture
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=self.num_final_ch, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(self.num_final_ch),
        )

        ## flattening of encoder output into linear layer
        self.embed_linear = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.num_final_ch * self.flat_dim, self.embed_dim)
        )

        ## bottleneck output to linear layer
        self.decode_linear = nn.Sequential(
            nn.Linear(self.embed_dim, self.num_final_ch * self.flat_dim),
            nn.ReLU(),
        )

        # decoder
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(in_channels=self.num_final_ch, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding="same")
        )

    def forward(self, x):
        # encoder
        x = self.encoder(x)
        #print(x.shape)
        # bottleneck output
        embedding = self.embed_linear(x)
        # bottleneck out to linear layer
        x = self.decode_linear(embedding)
        # reshape linear layer output to feed it to convolutional layers
        x = torch.reshape(x, (x.shape[0], self.num_final_ch, int(self.flat_dim**0.5), int(self.flat_dim**0.5)))
        # decoder
        x = self.decoder(x)
        return x, embedding

    def encode(self, x):
        x = self.encoder(x)
        embedding = self.embed_linear(x)
        return embedding

    def decode(self, embedding):
        x = self.decode_linear(embedding)
        x = torch.reshape(x, (x.shape[0], self.num_final_ch, int(self.flat_dim**0.5), int(self.flat_dim**0.5)))
        x = self.decoder(x)
        return x
