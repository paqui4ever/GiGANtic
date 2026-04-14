import torch
import torch.optim as optim
import torch.nn as nn 

from models import Generator, Discriminator

from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

# Define batch size and epoch num
BATCH_SIZE = 32
NUM_EPOCHS = 100

# Define device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Device both nn for GAN
discriminator = Discriminator(img_channels=1, dimension=28).to(device)
generator = Generator(latent_dim=50).to(device)

# Define optimizers for each nn
discriminator_optimizer = optim.AdamW(discriminator.parameters(), lr = 0.005, weight_decay=1e-5)
generator_optimizer = optim.AdamW(generator.parameters(), lr = 0.005, weight_decay=1e-5)

# Download the dataset and normalize between 0 and 1
train_dataset = MNIST(root="./data", download=True, train=True, transform=transforms.ToTensor()) # .ToTensor() normalizes between 0 and 1

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


def _discriminator_train_epoch(loader, batch_size, discriminator_optimizer, criterion):
    # Generate real and fake data
    generated_data = generator.sample(batch_size)
    real_data = loader.sample(batch_size)

    # Calculate discriminator's predictions for both real and fake
    probs_real = discriminator(real_data)
    probs_generated = discriminator(generated_data)

    # Compute the loss for the discriminator
    discriminator_optimizer.zero_grad()
    loss = probs_generated.mean() - probs_real.mean() # We want to maximize the difference between the real and the fake images

    # Backprop 
    loss.backward()
    discriminator_optimizer.step()

    # Weight clipping so that it satisfies the Lipschitz constraint (it limits how much the weights are changing per epoch)
    for p in critic.parameters(): 
        p.data.clamp_(-0.01, 0.01)

    return loss.item()

def _generator_train_epoch(batch_size, generator_optimizer, criterion):
    # Generate the data and calculate discriminators prediction
    generated_data = generator.sample(batch_size)
    probs_generated = discriminator(generated_data)

    generator_loss = -probs_generated.mean() # Its negative because we want to maximize the score for the discriminators score

    generator_optimizer.zero_grad()
    generator_loss.backward()
    generator_optimizer.step()

    return generator_loss.item()


print("Starting training...")

for epoch in range(NUM_EPOCHS):
    for batch_idx, img in enumerate(train_loader):
        _discriminator_train_epoch(train_loader, BATCH_SIZE, discriminator_optimizer, criterion)
        _generator_train_epoch(BATCH_SIZE, generator_optimizer, criterion)