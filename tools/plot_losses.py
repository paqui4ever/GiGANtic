import matplotlib.pyplot as plt
import os
import argparse
import wandb

def smooth_curve(points, factor=0.85):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

def create_plot(steps, vals, color, label, title, filename):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    ax.plot(steps, vals, alpha=0.4, color=color, label='_nolegend_', linewidth=0.5)
    ax.plot(steps, smooth_curve(vals), color=color, linewidth=2.5, label=label)
    
    fig.suptitle(title, fontsize=16, fontweight='bold', color='#333333', y=0.98, fontfamily='sans-serif')
    
    ax.set_xlabel('epoch', fontsize=13, labelpad=10, fontfamily='sans-serif', color='#555555')
    ax.set_ylabel('loss', fontsize=13, labelpad=10, fontfamily='sans-serif', color='#555555')
    
    ax.tick_params(axis='both', which='major', labelsize=11, colors='#555555')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=1, fontsize=12, frameon=False, labelspacing=0.5)
    
    ax.grid(axis='y', linestyle='-', alpha=0.4, color='#b0b0b0')
    ax.set_axisbelow(True)
    
    for spine in ax.spines.values():
        spine.set_edgecolor('#555555')
        spine.set_linewidth(1.0)
    
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    plt.subplots_adjust(top=0.9)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    plt.savefig(filename, bbox_inches='tight', format='svg')
    plt.close()
    print(f"Saved plot to {filename}")

def plot_wandb_losses(run_paths, output_prefix, title_prefix="GAN"):
    api = wandb.Api()
    
    all_gen_vals = []
    all_disc_vals = []
    
    for run_path in run_paths:
        print(f"Fetching run {run_path}...")
        try:
            run = api.run(run_path)
        except Exception as e:
            print(f"Failed to fetch run {run_path}: {e}")
            continue
            
        for row in run.scan_history(keys=['Loss/Generator', 'Loss/Discriminator']):
            if 'Loss/Generator' in row and row['Loss/Generator'] is not None:
                all_gen_vals.append(row['Loss/Generator'])
                
            if 'Loss/Discriminator' in row and row['Loss/Discriminator'] is not None:
                all_disc_vals.append(row['Loss/Discriminator'])
                
    if not all_gen_vals and not all_disc_vals:
        print("No loss data found in the provided runs.")
        return
        
    all_gen_steps = list(range(len(all_gen_vals)))
    all_disc_steps = list(range(len(all_disc_vals)))
    
    create_plot(all_gen_steps, all_gen_vals, '#1f77b4', 'Generator Loss', f'{title_prefix} Generator Loss', f'{output_prefix}_generator_loss.svg')
    create_plot(all_disc_steps, all_disc_vals, '#ff7f0e', 'Discriminator Loss', f'{title_prefix} Discriminator Loss', f'{output_prefix}_discriminator_loss.svg')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate modern SVG plots from wandb runs.")
    # --wandb_runs: The path(s) to the wandb run(s) you want to extract loss data from.
    # Example: "yourUserName/yourProjectName/yourRunID"
    # Note: Can accept multiple runs separated by spaces to concatenate their data.
    parser.add_argument("--wandb_runs", type=str, nargs='+', required=True, help="List of wandb run paths in order")
    
    # --output_prefix: The path and prefix for the output SVG files.
    # Example: "assets/EBGAN/EBGAN" will output "assets/EBGAN/EBGAN_generator_loss.svg"
    parser.add_argument("--output_prefix", type=str, required=True, help="Output path prefix for the generated SVGs")
    
    # --title_prefix: The text that will appear at the start of the title on the plots.
    # Example: "EBGAN" will create titles like "EBGAN Generator Loss"
    parser.add_argument("--title_prefix", type=str, required=True, help="Prefix for the plot titles")
    
    args = parser.parse_args()
    
    plot_wandb_losses(args.wandb_runs, args.output_prefix, args.title_prefix)