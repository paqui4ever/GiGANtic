import torch
import torch.optim as optim
import torch.nn as nn 

from models import Generator, Discriminator
from utils import gaussian_negative_log_likelihood

from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid

from torch.utils.data import DataLoader

import argparse

parser = argparse.ArgumentParser(description="Training script for GAN models")
parser.add_argument(
    "--model",
    type=str,
    choices=["WGAN", "DCGAN", "InfoGAN"],
    required=True,
    help="Model to train"
)
parser.add_argument(
    "--epochs",
    type=int,
    default=100,
    help="Number of epochs to train"
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=32,
    help="Batch size for training"
)

args = parser.parse_args()

# Define batch size and epoch num
BATCH_SIZE = args.batch_size # Will determine how precise is the expected value estimation
NUM_EPOCHS = args.epochs

# Define device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Device both nn for GAN
discriminator = Discriminator(img_channels=1, dimension=28).to(device)
generator = Generator(latent_dim=50).to(device)

# Define optimizers for each nn
if args.model == "DCGAN":
    # The weight decay of AdamW may hinder stability, favoring imbalance
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr = 0.0002, betas=((0.5, 0.999))) 
    generator_optimizer = optim.Adam(generator.parameters(), lr = 0.0002, betas=((0.5, 0.999)))
elif args.model == "WGAN":
    # Because this is a W-GAN with weight clipping, we use RMSprop instead of Adam, momentum will conflict with weight clipping
    discriminator_optimizer = optim.RMSprop(discriminator.parameters(), lr = 0.00005) 
    generator_optimizer = optim.RMSprop(generator.parameters(), lr = 0.00005)

# Download the dataset and normalize between 0 and 1
train_dataset = MNIST(root="./data", download=True, train=True, transform=transforms.ToTensor()) # .ToTensor() normalizes between 0 and 1

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

if args.model == "DCGAN":
    # We need to set up the Binary Cross Entropy Loss
    criterion = nn.BCEWithLogitsLoss() # Combines the sigmoid activation and the BCELoss in one single function using log-sum-exp trick

if args.model == "InfoGAN":
    criterion = nn.GaussianNLLLoss()

def _w_discriminator_train_epoch(loader, batch_size, discriminator_optimizer):
    # In the case of the W-GAN the discriminator measures the distance between the real and the fake distributions
    # Generate real and fake data
    generated_data = generator.sample(batch_size)
    real_data = loader.sample(batch_size)

    # Calculate discriminator's predictions for both real and fake
    probs_real = discriminator(real_data)
    probs_generated = discriminator(generated_data)

    # Compute the loss for the discriminator
    discriminator_optimizer.zero_grad()
    # We take the mean because, we know by Monte Carlo and the Law of Large Numbers, that as the number of samples n increases, the mean of the samples converges to the expected value
    # giving the Kantorovich-Rubinstein Duality (that's equal to the Wasserstein distance)
    loss = probs_generated.mean() - probs_real.mean() # We want to maximize the difference between the real and the fake images

    # Backprop 
    loss.backward()
    discriminator_optimizer.step()

    # Weight clipping so that it satisfies the Lipschitz constraint (it limits how much the weights are changing per epoch), therefore its 1-Lipschitz continuous
    for p in discriminator.parameters(): 
        p.data.clamp_(-0.01, 0.01)

    return loss.item()

def _w_generator_train_epoch(batch_size, generator_optimizer):
    # Generate the data and calculate discriminators prediction
    generated_data = generator.sample(batch_size)
    probs_generated = discriminator(generated_data)

    generator_loss = -probs_generated.mean() # Its negative because we want to maximize the score for the discriminators score

    generator_optimizer.zero_grad()
    generator_loss.backward()
    generator_optimizer.step()

    return generator_loss.item()

def _dc_discriminator_train_epoch(loader, batch_size, discriminator_optimizer, criterion):
    # Generate real and fake data
    generated_data = generator.sample(batch_size)
    real_data = loader.sample(batch_size)

    # Calculate discriminator's predictions for both real and fake
    probs_real = discriminator(real_data)
    probs_generated = discriminator(generated_data) 

    # Compute the loss for the discriminator
    discriminator_optimizer.zero_grad()
    # Trains the discriminator to output 1 for real images and 0 for fake images
    loss = criterion(probs_real, torch.ones_like(probs_real)) + criterion(probs_generated, torch.zeros_like(probs_generated))

    # Backprop
    loss.backward()
    discriminator_optimizer.step()

    return loss.item()


def _dc_generator_train_epoch(batch_size, generator_optimizer, criterion):
    generated_data = generator.sample(batch_size)
    probs_generated = discriminator(generated_data)

    # Trains the generator to see the generated images as real
    generator_loss = criterion(probs_generated, torch.ones_like(probs_generated))

    # Backprop
    generator_optimizer.zero_grad()
    generator_loss.backward()
    generator_optimizer.step()

    return generator_loss.item()

def _infogan_discriminator_train_epoch(loader, batch_size, discriminator_optimizer):
    pass

def _infogan_generator_train_epoch(batch_size, generator_optimizer, criterion, num_continuous_codes=2, lambda_var=1):
    # generated_data has the form [noise_z, codes]
    generated_data = generator.sample(batch_size, num_continuous_codes)
    probs_generated = discriminator(generated_data)

    discrete_logits, mean, log_var = Q_head(generated_data)

    # Compute the discrete loss
    discrete_loss = nn.CrossEntropyLoss(discrete_logits, torch.ones_like(discrete_logits))

    # Compute the continuous loss
    var = torch.exp(log_var)
    codes = generated_data[:, batch_size:]
    continuous_loss = criterion(codes, mean, var=var)

    # continuous_loss = gaussian_negative_log_likelihood(codes, mean, var)
    
    # Trains the generator to see the generated images as real
    generator_loss = criterion(probs_generated, torch.ones_like(probs_generated))

    # Backprop
    generator_optimizer.zero_grad()
    generator_loss.backward()
    generator_optimizer.step()

    return generator_loss.item()


print("Starting training...")

discriminator_step_number = 0

# Fix latents to evaluate progress via gif 

fixed_latents = (generator.sample(num_samples=64)).to(DEVICE)
progress_images = []

# Training loop
for epoch in range(NUM_EPOCHS):
    for batch_idx, img in enumerate(train_loader):

        if args.model == "WGAN":
            _w_discriminator_train_epoch(train_loader, BATCH_SIZE, discriminator_optimizer)

            discriminator_step_number += 1
            if discriminator_step_number % 5 == 0: # We ensure the critic is "good enough" (closer to D*, that being optimal D) before updating the generator
                _w_generator_train_epoch(BATCH_SIZE, generator_optimizer)
        elif args.model == "DCGAN":
            _dc_discriminator_train_epoch(train_loader, BATCH_SIZE, discriminator_optimizer, criterion)
            _dc_generator_train_epoch(BATCH_SIZE, generator_optimizer, criterion)
        elif args.model == "InfoGAN":
            _infogan_discriminator_train_epoch(train_loader, BATCH_SIZE, discriminator_optimizer)
            _infogan_generator_train_epoch(BATCH_SIZE, generator_optimizer)

        if epoch % 5 == 0:
            image_grid = make_grid(generator(fixed_latents).cpu())
            # Convert to numpy + format to (H, W, C)
            image_grid = np.transpose(image_grid.numpy, (1,2,0))

            progress_images.append(image_grid)
    
    if epoch % 5 == 0:
        imageio.mimsave(f'./training_{epoch}_epoch.gif', progress_images)