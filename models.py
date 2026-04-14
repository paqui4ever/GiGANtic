import torch 
import torch.nn as nn

class Generator(nn.Module):
    # number_of_generator_features defaults to 28 because of MNIST dataset 28x28 images
    # img_channels defaults to 1 because of MNIST dataset grayscale images
    def __init__(self, latent_dim, number_of_generator_features=28, img_channels=1):
        super().__init__()
        self.latent_dim = latent_dim

        self.net = nn.Sequential(
            # First layer acts as a linear layer, scaling the latent vector to a 4x4 image with number_of_generator_features*8 channels
            _block(latent_dim, number_of_generator_features*8, kernel_size=4, stride=1, padding=0),

            _block(number_of_generator_features*8, number_of_generator_features*4, kernel_size=4, stride=2, padding=1),
            
            _block(number_of_generator_features*4, number_of_generator_features*2, kernel_size=4, stride=2, padding=1),
            
            nn.ConvTranspose2d(number_of_generator_features*2, img_channels, kernel_size=4, stride=2, padding=1),

            nn.Tanh() # Using tanh to squeeze the pixel values between [-1, 1]
        )

    def _block(in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            # No bias because of BatchNorm after the convolution
            nn.BatchNorm2d(out_channels),
            # Leaky ReLU allows for a small alpha*activation when the activation is <0
            nn.LeakyReLU(inplace=True) # inplace=True doesn't make a copy and is more memory efficient
        )

    def forward(self, x):
        # If input is 2D being (Batch, latent_dim), we need to make it 4D being (Batch, latent_dim, 1, 1),
        # basically making it a 1x1 image with latent_dim number of channels

        if len(x.shape) == 2:
            x = x.view(x.shape[0], self.latent_dim, 1, 1)

        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, img_channels, dimension):
        super().__init__()

        self.net = nn.Sequential(
            _block(img_channels, dimension*2, kernel_size=4, stride=2, padding=1),

            _block(dimension*2, dimension*4, kernel_size=4, stride=2, padding=1),

            _block(dimension*4, dimension*8, kernel_size=4, stride=2, padding=1),

            nn.Conv2d(dimension*8, 1, kernel_size=4, stride=1, padding=0),

            # nn.Sigmoid() # Using sigmoid to squeeze the pixel values between [0, 1]
        )

    def _block(in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            # No bias because of BatchNorm after the convolution
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True) # inplace=True doesn't make a copy and is more memory efficient
        )

    def forward(self, x):
        x = self.net(x)
        return x.view(x.shape[0], -1) # Flatten the output to a 1D vector
