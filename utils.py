import torch
import matplotlib.pyplot as plt
import os
import glob
from tensorboard.backend.event_processing import event_accumulator

def gaussian_negative_log_likelihood(codes, mu, log_var):
    squared_error = (codes - mu) ** 2
    # torch.exp(-logvar) = 1 / sigma**2
    weighted_squared_error = squared_error * torch.exp(-log_var)

    penalty_term = 0.5 * log_var

    return weighted_squared_error + penalty_term

def smooth_curve(points, factor=0.85):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

def plot_gan_losses(run_dir, output_prefix, title_prefix="GAN"):
    """
    Reads TensorBoard events from run_dir and plots Generator and Discriminator losses.
    The plots are saved to output_prefix + _generator.png and _discriminator.png.
    """
    event_files = glob.glob(os.path.join(run_dir, 'events.out.tfevents.*'))
    if not event_files:
        print(f"No event files found in {run_dir}")
        return
    
    event_file = event_files[0]
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()
    
    if 'Loss/Generator' not in ea.Tags()['scalars'] or 'Loss/Discriminator' not in ea.Tags()['scalars']:
        print(f"Required tags not found in {run_dir}")
        return
        
    gen_events = ea.Scalars('Loss/Generator')
    disc_events = ea.Scalars('Loss/Discriminator')
    
    gen_steps = [e.step for e in gen_events]
    gen_vals = [e.value for e in gen_events]
    
    disc_steps = [e.step for e in disc_events]
    disc_vals = [e.value for e in disc_events]
    
    def create_plot(steps, vals, color, label, title, filename):
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        
        # Plot raw data (faded)
        ax.plot(steps, vals, alpha=0.25, color=color, label='_nolegend_')
        
        # Plot smoothed data
        ax.plot(steps, smooth_curve(vals), color=color, linewidth=2.5, label=label)
        
        # Formatting
        ax.set_title(title, fontsize=18, fontweight='bold', pad=20, fontfamily='sans-serif')
        ax.set_xlabel('Epoch', fontsize=14, labelpad=15, fontfamily='sans-serif')
        ax.set_ylabel('Loss', fontsize=14, labelpad=15, fontfamily='sans-serif')
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.legend(loc='upper right', fontsize=12, frameon=True, shadow=True, fancybox=True)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        ax.set_facecolor('#fafafa')
        fig.patch.set_facecolor('white')
        
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        print(f"Saved plot to {filename}")

    # Create Generator Plot
    create_plot(gen_steps, gen_vals, '#1f77b4', 'Generator Loss', f'{title_prefix} Generator Loss', f'{output_prefix}_generator_loss.png')
    
    # Create Discriminator Plot
    create_plot(disc_steps, disc_vals, '#ff7f0e', 'Discriminator Loss', f'{title_prefix} Discriminator Loss', f'{output_prefix}_discriminator_loss.png')

def generate_all_plots():
    plot_gan_losses('./runs/DCGAN_training', './assets/DCGAN', 'DCGAN')
    plot_gan_losses('./runs/InfoGAN_training', './assets/InfoGAN', 'InfoGAN')
    plot_gan_losses('./runs/WGAN_training', './assets/WGAN', 'WGAN')
    # plot_gan_losses('./runs/WGAN-GP_training', './assets/WGAN-GP', 'WGAN-GP')
