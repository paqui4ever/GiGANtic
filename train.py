import torch
import torch.optim as optim
import torch.nn as nn 

from models import Generator, Discriminator, WDiscriminator, InfoGenerator, Q_head, EBDiscriminator, GeneratorWithBatchNorm
from utils import gaussian_negative_log_likelihood

from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import argparse
import wandb
import datetime
import imageio
import os
import numpy as np
import random

from tqdm import tqdm

import itertools

parser = argparse.ArgumentParser(description="Training script for GAN models")
parser.add_argument(
    "--model",
    type=str,
    choices=["WGAN", "DCGAN", "InfoGAN", "EBGAN"],
    required=True,
    help="Model to train"
)
parser.add_argument(
    "--epochs",
    type=int,
    required=True,
    help="Number of epochs to train"
)
parser.add_argument(
    "--batch_size",
    type=int,
    required=True,
    help="Batch size for training"
)
parser.add_argument(
    "--save_image_every_n_epochs",
    type=int,
    default=15,
    help="Save an image every n epochs"
)
parser.add_argument(
    "--resume_checkpoint",
    type=int,
    default=0,
    help="Epoch to resume training from. Default is 0 (start from scratch)."
)
parser.add_argument(
    "--checkpoint_dir",
    type=str,
    required=True,
    help="Directory to save checkpoints"
)
parser.add_argument(
    "--gradient-penalty",
    action="store_true",
    help="Use gradient penalty on W-GANs discriminator"
)
parser.add_argument(
    "--save-gif",
    action="store_true",
    help="Save a gif of the generated images every n epochs"
)
parser.add_argument(
    "--amp",
    action="store_true",
    help="Enable Automatic Mixed Precision (AMP)"
)
parser.add_argument(
    "--seed",
    type=int,
    default=24,
    help="Random seed for reproducibility"
)

args = parser.parse_args()

if args.seed is not None:
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)
    print(f"Set random seed to {args.seed}")

# Define batch size and epoch num
BATCH_SIZE = args.batch_size # Will determine how precise is the expected value estimation
NUM_EPOCHS = args.epochs

# Define device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# The scaler is needed so gradients can be in the float16 range 
scaler_d = torch.amp.GradScaler(DEVICE, enabled=args.amp) if args.amp and DEVICE == 'cuda' else None
scaler_g = torch.amp.GradScaler(DEVICE, enabled=args.amp) if args.amp and DEVICE == 'cuda' else None

CHECKPOINT_DIR = args.checkpoint_dir
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Define both nn for GAN (and optimizers in the case of InfoGAN)
if args.model == "WGAN":
    discriminator = WDiscriminator(img_channels=1, dimension=28).to(DEVICE)
elif args.model == "EBGAN":
    discriminator = EBDiscriminator(channels=1, hidden_dim=64).to(DEVICE)
else:
    discriminator = Discriminator(img_channels=1, dimension=28).to(DEVICE)

q_head = None

if args.model == "InfoGAN":
    generator = InfoGenerator(latent_dim=50).to(DEVICE)
    # The feature extractor outputs a 28*8 feature map because the dimension is 28 and the last layer has 8 times the number of dimensions
    q_head = Q_head(feature_channels=28 * 8).to(DEVICE)
    # itertools.chain bundles its parameters together
    generator_optimizer = optim.Adam(itertools.chain(generator.parameters(), q_head.parameters()), lr = 1e-3, betas=((0.5, 0.999)))
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr = 2e-4, betas=((0.5, 0.999))) 
elif args.model in ["EBGAN", "DCGAN"]:
    generator = GeneratorWithBatchNorm(latent_dim=50).to(DEVICE)
else:
    # The normal Generator class doesn't have batchnorm
    generator = Generator(latent_dim=50).to(DEVICE)

# Define optimizers for each nn
if args.model == "DCGAN":
    # The weight decay of AdamW may hinder stability, favoring imbalance
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr = 2e-4, betas=((0.5, 0.999))) 
    generator_optimizer = optim.Adam(generator.parameters(), lr = 1e-3, betas=((0.5, 0.999)))
elif args.model == "WGAN":
    # Because this is a W-GAN with weight clipping, we use RMSprop instead of Adam, momentum will conflict with weight clipping
    discriminator_optimizer = optim.RMSprop(discriminator.parameters(), lr = 5e-5) 
    generator_optimizer = optim.RMSprop(generator.parameters(), lr = 5e-5)
elif args.model == "EBGAN":
    # Following the hyperparameters set in the original paper
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr = 2e-4, betas=((0.5, 0.999)))
    generator_optimizer = optim.Adam(generator.parameters(), lr = 2e-4, betas=((0.5, 0.999)))

# Download the dataset and normalize between 0 and 1
train_dataset = MNIST(root="./data", download=True, train=True, transform=transforms.Compose([
    transforms.Resize(32), # Pad to not break Discriminator and Generator logic of pooling
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) # Shifts from [0,1] to [-1,1]
])) # .ToTensor() normalizes between 0 and 1

# Define the train loader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Setting up criterions for models that need them
if args.model == "DCGAN":
    # We need to set up the Binary Cross Entropy Loss
    criterion = nn.BCEWithLogitsLoss() # Combines the sigmoid activation and the BCELoss in one single function using log-sum-exp trick
elif args.model == "InfoGAN":
    criterion = nn.BCEWithLogitsLoss() # Combines the sigmoid activation and the BCELoss in one single function using log-sum-exp trick
    continuous_criterion = nn.GaussianNLLLoss()
    discrete_criterion = nn.CrossEntropyLoss()
elif args.model == "EBGAN":
    margin = 0.2 # It must be less than 4 because its the maximum possible difference between to values in [-1,1] range
               # Then, the max squared difference is 4
    criterion = nn.MSELoss()

# Define private functions to train each model for one epoch
def _w_discriminator_train_epoch(real_data, batch_size, discriminator_optimizer):
    # In the case of the W-GAN the discriminator (more often called critic in this case) measures the distance between the real 
    # and the fake distributions
    discriminator_optimizer.zero_grad()
    
    with torch.amp.autocast(device_type=DEVICE, enabled=args.amp):
        # Generate fake data out of the latent
        latent = generator.sample(batch_size, device=DEVICE)
        generated_data = generator(latent)
        # real_data = loader.sample(batch_size)

        # Calculate discriminator's predictions for both real and fake
        probs_real, _ = discriminator(real_data)
        probs_generated, _ = discriminator(generated_data.detach()) # detach so it doesnt ruin the generators weights

        # Compute the loss for the discriminator
        # We take the mean because, we know by Monte Carlo and the Law of Large Numbers, that as the number of samples n increases, the mean of the samples converges to the expected value
        # giving the Kantorovich-Rubinstein Duality (that's equal to the Wasserstein distance)
        loss = probs_generated.mean() - probs_real.mean() # We want to maximize the difference between the real and the fake images

    # Backprop 
    if scaler_d:
        # Scale the gradients to fp16 and backward guaranteeing no underflow 
        scaler_d.scale(loss).backward()
        # Unscales + checks if theres any NaN (if there is it skips step)
        scaler_d.step(discriminator_optimizer)
        # Updates the scale for the next iteration trying to increase it after some time being stable (no NaNs)
        scaler_d.update()
    else:
        loss.backward()
        discriminator_optimizer.step()

    # Weight clipping so that it satisfies the Lipschitz constraint (it limits how much the weights are changing per epoch), therefore its 1-Lipschitz continuous
    for p in discriminator.parameters(): 
        p.data.clamp_(-0.01, 0.01)

    return loss.item()

def _w_generator_train_epoch(batch_size, generator_optimizer):
    generator_optimizer.zero_grad()
    
    with torch.amp.autocast(device_type=DEVICE, enabled=args.amp):
        # Generate the data and calculate discriminators prediction
        latent = generator.sample(batch_size, device=DEVICE)
        generated_data = generator(latent)
        probs_generated, _ = discriminator(generated_data)

        generator_loss = -probs_generated.mean() # Its negative because we want to maximize the score for the discriminators score

    # Backprop
    if scaler_g:
        scaler_g.scale(generator_loss).backward()
        scaler_g.step(generator_optimizer)
        scaler_g.update()
    else:
        generator_loss.backward()
        generator_optimizer.step()

    return generator_loss.item()

def _dc_discriminator_train_epoch(real_data, batch_size, discriminator_optimizer, criterion):
    discriminator_optimizer.zero_grad()
    
    with torch.amp.autocast(device_type=DEVICE, enabled=args.amp):
        # Generate real and fake data
        latent = generator.sample(batch_size, device=DEVICE)
        generated_data = generator(latent)
        # real_data = loader.sample(batch_size)

        # Calculate discriminator's predictions for both real and fake
        probs_real, _ = discriminator(real_data)
        probs_generated, _ = discriminator(generated_data.detach()) # detach so it doesnt ruin the generators weights by flowing back through generated_data

        # Compute the loss for the discriminator
        # Trains the discriminator to output 1 for real images and 0 for fake images, that being, maximizing the probability of correctly labeling the images
        loss = criterion(probs_real, torch.ones_like(probs_real)) + criterion(probs_generated, torch.zeros_like(probs_generated))

    # Backprop
    if scaler_d:
        scaler_d.scale(loss).backward()
        scaler_d.step(discriminator_optimizer)
        scaler_d.update()
    else:
        loss.backward()
        discriminator_optimizer.step()

    return loss.item()


def _dc_generator_train_epoch(batch_size, generator_optimizer, criterion):
    generator_optimizer.zero_grad()
    
    with torch.amp.autocast(device_type=DEVICE, enabled=args.amp):
        latent = generator.sample(batch_size, DEVICE)
        generated_data = generator(latent)
        probs_generated, _ = discriminator(generated_data)

        # Trains the discriminator to see the generated images as real, minimizing the loss between the discriminator's prediction and 1
        generator_loss = criterion(probs_generated, torch.ones_like(probs_generated))

    # Backprop
    if scaler_g:
        scaler_g.scale(generator_loss).backward()
        scaler_g.step(generator_optimizer)
        scaler_g.update()
    else:
        generator_loss.backward()
        generator_optimizer.step()

    return generator_loss.item()

# def _infogan_discriminator_train_epoch(loader, batch_size, discriminator_optimizer):
#     pass

def _infogan_generator_train_epoch(batch_size, generator_optimizer, criterion, num_continuous_codes=2, lambda_var=1):
    generator_optimizer.zero_grad()
    
    with torch.amp.autocast(device_type=DEVICE, enabled=args.amp):
        # generated_data has the form [noise_z, codes]
        latent = generator.sample(batch_size, device=DEVICE, num_continuous_codes=num_continuous_codes)
        generated_data = generator(latent)
        probs_generated, features = discriminator(generated_data)

        discrete_logits, mean, log_var = q_head(features)

        # Compute the discrete loss via CE
        discrete_loss = discrete_criterion(discrete_logits, torch.ones_like(discrete_logits))

        # Compute the continuous loss via GNLL
        var = torch.exp(log_var)
        codes = latent[:, -num_continuous_codes:]
        continuous_loss = continuous_criterion(mean, codes, var=var)

        # continuous_loss = gaussian_negative_log_likelihood(codes, mean, var)
        
        # Trains the generator to see the generated images as real
        generator_loss = criterion(probs_generated, torch.ones_like(probs_generated))

        # Complete the loss computation
        codes_loss = continuous_loss + discrete_loss

        # Loss follows V(D, G) - lambda * L_I(G, Q) where L_I is the lower bound of the mutual information
        complete_gen_loss = generator_loss + lambda_var * codes_loss

    # Backprop
    if scaler_g:
        scaler_g.scale(complete_gen_loss).backward()
        scaler_g.step(generator_optimizer)
        scaler_g.update()
    else:
        complete_gen_loss.backward()
        generator_optimizer.step()

    return complete_gen_loss.item()

def _eb_discriminator_train_epoch(real_data, batch_size, discriminator_optimizer, criterion):
    discriminator_optimizer.zero_grad()

    with torch.amp.autocast(device_type=DEVICE, enabled=args.amp):
        latents = generator.sample(batch_size, device=DEVICE)
        generated_data = generator(latents).detach()

        # The discriminator is an autoencoder so it's trying to "reconstruct" the input image
        real_reconstruction = discriminator(real_data)
        # Must output low energy, so that it is close to the data manifold
        real_energy = criterion(real_reconstruction, real_data)

        fake_reconstruction = discriminator(generated_data)
        # Must output high energy (but close to margin) to push further of the data manifold
        fake_energy = criterion(fake_reconstruction, generated_data)

        # Use ReLU margin to keep the second term positive in case that fake_energy > margin
        # The margin serves a limit to how high the fake_energy can be, so when a fake sample has energy high enough, the loss for it will be 0
        # That way it lets D focus on other samples and doesn't let D output inf energy destroying gradients for fake images
        loss_discriminator = real_energy + torch.relu(margin - fake_energy)

    if scaler_d:
        scaler_d.scale(loss_discriminator).backward()
        scaler_d.step(discriminator_optimizer)
        scaler_d.update()
    else:
        loss_discriminator.backward()
        discriminator_optimizer.step()

    return loss_discriminator.item()

def _eb_generator_train_epoch(batch_size, generator_optimizer, criterion):
    generator_optimizer.zero_grad()

    with torch.amp.autocast(device_type=DEVICE, enabled=args.amp):
        latents = generator.sample(batch_size, device=DEVICE)
        generated_data = generator(latents)
        
        fake_reconstruction = discriminator(generated_data)
        loss_generator = criterion(fake_reconstruction, generated_data)

    if scaler_g:
        scaler_g.scale(loss_generator).backward()
        scaler_g.step(generator_optimizer)
        scaler_g.update()
    else:
        loss_generator.backward()
        generator_optimizer.step()

    return loss_generator.item()


# Manage checkpoint loading 

if args.resume_checkpoint > 0:
    generator.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/{args.model}_generator_epoch_{args.resume_checkpoint}.pth", map_location=DEVICE))
    discriminator.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/{args.model}_discriminator_epoch_{args.resume_checkpoint}.pth", map_location=DEVICE))
    print(f"Loaded checkpoint for epoch {args.resume_checkpoint}")

print(f"Starting training of {args.model} on {DEVICE}...")

# Count trainable parameters
generator_num_params = sum(p.numel()for p in generator.parameters() if p.requires_grad)
discriminator_num_params = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)

print(f"The current models generator has {generator_num_params} and the discriminator has {discriminator_num_params} (all trainable) ")

discriminator_step_number = 0

# Fix latents to evaluate progress via gif 
latents_path = f"{CHECKPOINT_DIR}/{args.model}_fixed_latents.pt"
if args.resume_checkpoint > 0 and os.path.exists(latents_path):
    fixed_latents = torch.load(latents_path, map_location=DEVICE)
    print("Loaded fixed latents from checkpoint")
else:
    fixed_latents = (generator.sample(num_samples=64, num_continuous_codes=2, device=DEVICE)).to(DEVICE)
    torch.save(fixed_latents, latents_path)
    print("Saved new fixed latents")

progress_images = []

# Tensorboard writer to log losses and images
# writer = SummaryWriter(log_dir=f'./runs/{args.model}_training')

# Setup config dictionary to include optimizer and criterion info
config_dict = vars(args).copy()
config_dict["generator_optimizer"] = generator_optimizer.__class__.__name__
config_dict["discriminator_optimizer"] = discriminator_optimizer.__class__.__name__

if args.model in ["DCGAN", "EBGAN"]:
    config_dict["criterion"] = criterion.__class__.__name__
elif args.model == "InfoGAN":
    config_dict["criterion"] = criterion.__class__.__name__
    config_dict["continuous_criterion"] = continuous_criterion.__class__.__name__
    config_dict["discrete_criterion"] = discrete_criterion.__class__.__name__
elif args.model == "WGAN":
    config_dict["criterion"] = "Wasserstein Distance"

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Initialize wandb
wandb.init(
    project="GiGANtic",
    name=f"{args.model}_training_{timestamp}",
    config=config_dict
)

# Training loop
for epoch in tqdm(range(args.resume_checkpoint, NUM_EPOCHS), desc="Epochs"):
    epoch_d_loss = 0.0
    epoch_g_loss = 0.0
    wgan_g_steps = 0

    pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False)
    for batch_idx, (imgs, _) in enumerate(pbar):
        imgs = imgs.to(DEVICE)

        if args.model == "WGAN":
            d_loss = _w_discriminator_train_epoch(imgs, BATCH_SIZE, discriminator_optimizer)
            epoch_d_loss += d_loss

            discriminator_step_number += 1
            if discriminator_step_number % 5 == 0: # We ensure the critic is "good enough" (closer to D*, that being optimal D) before updating the generator
                g_loss = _w_generator_train_epoch(BATCH_SIZE, generator_optimizer)
                epoch_g_loss += g_loss
                wgan_g_steps += 1
        elif args.model == "DCGAN":
            d_loss = _dc_discriminator_train_epoch(imgs, BATCH_SIZE, discriminator_optimizer, criterion)
            g_loss = _dc_generator_train_epoch(BATCH_SIZE, generator_optimizer, criterion)
            epoch_d_loss += d_loss
            epoch_g_loss += g_loss
        elif args.model == "InfoGAN":
            # The discriminator has the same max_D V(D, G) objective as DCGAN
            d_loss = _dc_discriminator_train_epoch(imgs, BATCH_SIZE, discriminator_optimizer, criterion)
            g_loss = _infogan_generator_train_epoch(BATCH_SIZE, generator_optimizer, criterion)
            epoch_d_loss += d_loss
            epoch_g_loss += g_loss
        elif args.model == "EBGAN":
            d_loss = _eb_discriminator_train_epoch(imgs, BATCH_SIZE, discriminator_optimizer, criterion)
            g_loss = _eb_generator_train_epoch(BATCH_SIZE, generator_optimizer, criterion)
            epoch_d_loss += d_loss
            epoch_g_loss += g_loss

        # Make the imagegrid for the gifs
        if args.save_gif:
            if epoch % args.save_image_every_n_epochs == 0:
                with torch.no_grad():
                    image_grid_tensor = make_grid(generator(fixed_latents).cpu(), normalize=True, value_range=(-1, 1))
                    image_grid_np = (np.transpose(image_grid_tensor.numpy(), (1, 2, 0)) * 255).astype(np.uint8)
                    progress_images.append(image_grid_np)
    
    # Save the imagegrid for the gifs
    if args.save_gif:
        if epoch % args.save_image_every_n_epochs == 0:
            imageio.mimsave(f'./gifs/{args.model}_training_{epoch}_epoch.gif', progress_images)
            progress_images = []

    avg_d_loss = epoch_d_loss / len(train_loader)
    avg_g_loss = (epoch_g_loss / wgan_g_steps) if (args.model == "WGAN" and wgan_g_steps > 0) else (epoch_g_loss / len(train_loader))
    
    # Log the losses on the tensorboard and print to screen
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] D_Loss: {avg_d_loss:.4f} G_Loss: {avg_g_loss:.4f}")
    # writer.add_scalar("Loss/Discriminator", avg_d_loss, epoch)
    # writer.add_scalar("Loss/Generator", avg_g_loss, epoch)

    wandb_log_dict = {
        "Loss/Discriminator": avg_d_loss,
        "Loss/Generator": avg_g_loss,
        "epoch": epoch
    }

    # Checkpoint every 25 epochs
    if (epoch + 1) % 25 == 0:
        torch.save(generator.state_dict(), f"{CHECKPOINT_DIR}/{args.model}_generator_epoch_{epoch+1}.pth")
        torch.save(discriminator.state_dict(), f"{CHECKPOINT_DIR}/{args.model}_discriminator_epoch_{epoch+1}.pth")
        print(f"Checkpoints saved to {CHECKPOINT_DIR} at epoch {epoch+1}")

    # Save images to tensorboard every 15 epochs
    if epoch % args.save_image_every_n_epochs == 0:
        with torch.no_grad():
            image_grid = make_grid(generator(fixed_latents).cpu(), normalize=True, value_range=(-1, 1))
            # writer.add_image("Generated Images", image_grid, global_step=epoch)
            
            image_grid_np = (np.transpose(image_grid.numpy(), (1, 2, 0)) * 255).astype(np.uint8)
            wandb_log_dict["Generated Images"] = wandb.Image(image_grid_np, caption=f"Epoch {epoch}")

    wandb.log(wandb_log_dict, step=epoch)

# writer.close()
wandb.finish()

