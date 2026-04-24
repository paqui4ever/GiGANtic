import torch 
import torch.nn as nn

class Generator(nn.Module):
    # number_of_generator_features defaults to 28 because of MNIST dataset 28x28 images
    # img_channels defaults to 1 because of MNIST dataset grayscale images
    def __init__(self, latent_dim=62, number_of_generator_features=28, img_channels=1):
        super().__init__()
        self.latent_dim = latent_dim

        self.net = nn.Sequential(
            # First layer acts as a linear layer, scaling the latent vector to a 4x4 image with number_of_generator_features*8 channels
            self._block(latent_dim, number_of_generator_features*8, kernel_size=4, stride=1, padding=0),

            self._block(number_of_generator_features*8, number_of_generator_features*4, kernel_size=4, stride=2, padding=1),
            
            self._block(number_of_generator_features*4, number_of_generator_features*2, kernel_size=4, stride=2, padding=1),
            
            nn.ConvTranspose2d(number_of_generator_features*2, img_channels, kernel_size=4, stride=2, padding=1),

            nn.Tanh() # Using tanh to squeeze the pixel values between [-1, 1]
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            # No bias because of BatchNorm after the convolution
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True) # inplace=True doesn't make a copy and is more memory efficient
        )

    def forward(self, x):
        # If input is 2D being (Batch, latent_dim), we need to make it 4D being (Batch, latent_dim, 1, 1),
        # basically making it a 1x1 image with latent_dim number of channels

        if len(x.shape) == 2:
            x = x.view(x.shape[0], x.shape[1], 1, 1)

        return self.net(x)

    def sample(self, num_samples, device, **kwargs):
        return torch.randn((num_samples, self.latent_dim), device=device)

class InfoGenerator(Generator):
    def __init__(self, latent_dim=62, number_of_generator_features=28, img_channels=1, num_continuous_codes=2):
        # Initialize taking into account the continuos codes added
        super().__init__(latent_dim + num_continuous_codes, number_of_generator_features, img_channels)
        self.noise_dim = latent_dim

    def sample(self, num_samples, device, num_continuous_codes=2, **kwargs):
        noise_z = torch.randn((num_samples, self.noise_dim), device=device)

        # Rescaling following (high - low) * torch.rand(size) + low so that is in range [-1, 1]
        codes = 2 * torch.rand((num_samples, num_continuous_codes), device=device) - 1

        # Now the generator will get fed G(z, c) instead of only G(z)
        return torch.cat([noise_z, codes], dim=1)

class Discriminator(nn.Module):
    def __init__(self, img_channels, dimension):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            self._block(img_channels, dimension*2, kernel_size=4, stride=2, padding=1),

            self._block(dimension*2, dimension*4, kernel_size=4, stride=2, padding=1),

            self._block(dimension*4, dimension*8, kernel_size=4, stride=2, padding=1),

            # nn.Sigmoid() # Using sigmoid to squeeze the pixel values between [0, 1]
        )

        self.D_head = nn.Conv2d(dimension*8, 1, kernel_size=4, stride=1, padding=0)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            # No bias because of BatchNorm after the convolution
            # nn.BatchNorm2d(out_channels),
            # Leaky ReLU allows for a small alpha*activation when the activation is <0 (for W-GANs, when a sample is classified as being very fake)
            # Basically, it attacks the dying ReLU problem
            nn.LeakyReLU(0.2, inplace=True) 
        )

    def forward(self, x):
        features = self.feature_extractor(x) # We need to return the features so they can be fed to the Q_head in the InfoGAN
        validity = self.D_head(features)
        return validity.view(validity.shape[0], -1), features # Flatten the output to a 1D vector

class WDiscriminator(Discriminator):
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        # No batchnorm because it changes the output of an image based on others, so that ruins the Wasserstein distance calculation
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            # Leaky ReLU allows for a small alpha*activation when the activation is <0 (for W-GANs, when a sample is classified as being very fake)
            # nn.InstanceNorm2d(out_channels), # Swap batchnorm for instance norm so it doesn't affect training
            # Basically, it attacks the dying ReLU problem
            nn.LeakyReLU(0.2, inplace=True) 
        )

class Q_head(nn.Module):
    # For InfoGAN the Q_head must return the mean (mu) and the variance (sigma squared) because we asume that 
    # the relationship between the code and the image in Q(c|x) follows a Gaussian normal distribution

    def __init__(self, feature_channels, num_discrete_codes=10, num_continuous_codes=2):
        super().__init__()

        # safely pool the 3x3 or 4x4 feature map down to 1x1
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(feature_channels, 128)

        self.activation = nn.LeakyReLU(0.2, inplace=True)

        # Discrete logit branch that will learn a categorical distribution
        self.discrete_logits = nn.Linear(128, num_discrete_codes)

        # Continuous mean and log-variance branch that will learn a Gaussian distribution
        self.mean = nn.Linear(128, num_continuous_codes)
        # Using log variance we allow the input to be negative and the output to be in (-inf, +inf)
        self.log_var = nn.Linear(128, num_continuous_codes)

    def forward(self, features):
        x = self.pool(features)
        x = x.view(x.shape[0], -1) # Flatten the features that were of shape (Batch, feature_channels, 1, 1) to (Batch, 1)
        
        x = self.fc(self.activation(x))

        discrete_logits = self.discrete_logits(x)

        mean = self.mean(x)
        log_var = self.log_var(x)

        return discrete_logits, mean, log_var