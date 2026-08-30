import argparse
import wandb
import os
import re
import imageio

def generate_gif_from_wandb(initial_run_path, download_path, output_gif):
    api = wandb.Api()
    
    run_paths = [initial_run_path]
    
    while True:
        choice = input("Do you want to add images from another wandb run? (y/n): ").strip().lower()
        if choice == 'y':
            extra_run = input("Enter the wandb run path (e.g., 'entity/project/run_id'): ").strip()
            if extra_run:
                run_paths.append(extra_run)
        elif choice == 'n':
            break
        else:
            print("Please answer 'y' or 'n'.")
    
    os.makedirs(download_path, exist_ok=True)
    all_image_files = []
    
    for run_path in run_paths:
        print(f"\nFetching run: {run_path}")
        try:
            run = api.run(run_path)
        except Exception as e:
            print(f"Error fetching run from wandb: {e}")
            continue

        run_id = run_path.split("/")[-1]
        run_download_path = os.path.join(download_path, run_id)
        os.makedirs(run_download_path, exist_ok=True)
        
        print(f"Downloading images to: {run_download_path}")
        run_image_files = []
        
        # In wandb, images are usually saved in the 'media/images/' directory.
        for file in run.files():
            if file.name.startswith("media/images/") and (file.name.endswith(".png") or file.name.endswith(".jpg")):
                file.download(root=run_download_path, replace=True)
                run_image_files.append(os.path.join(run_download_path, file.name))
                
        if not run_image_files:
            print(f"No images found in run: {run_path}")
            continue
            
        print(f"Downloaded {len(run_image_files)} images from {run_path}.")
        
        # Sort files to ensure they are in chronological order
        def get_step(filename):
            basename = os.path.basename(filename)
            match = re.search(r'_(\d+)_[a-f0-9]+\.(png|jpg)$', basename)
            if match:
                return int(match.group(1))
            match = re.search(r'_(\d+)\.(png|jpg)$', basename)
            if match:
                return int(match.group(1))
            numbers = re.findall(r'\d+', basename)
            if numbers:
                return int(numbers[-1])
            return 0

        try:
            run_image_files.sort(key=get_step)
        except Exception as e:
            print(f"Sorting by step failed for {run_path}, using default sort: {e}")
            run_image_files.sort()
            
        all_image_files.extend(run_image_files)
        
    if not all_image_files:
        print("No valid images found to create GIF.")
        return
        
    print(f"\nGenerating GIF at: {output_gif} with {len(all_image_files)} total frames...")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(os.path.abspath(output_gif))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    images = []
    # Use imageio.v2.imread if available to avoid deprecation warnings
    imread = getattr(imageio, 'v2', imageio).imread
    
    for filename in all_image_files:
        try:
            images.append(imread(filename))
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            
    if images:
        fps = 5
        # Make the last image last for 1 entire second
        images.extend([images[-1]] * (fps - 1))
        
        imageio.mimsave(output_gif, images, fps=fps, loop=0)
        print(f"GIF successfully generated: {output_gif}")
    else:
        print("No valid images to create GIF.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a GIF from images logged to a wandb run.")
    parser.add_argument("--run_path", type=str, required=True, help="Wandb run path (e.g., 'entity/project/run_id')")
    parser.add_argument("--download_path", type=str, required=True, help="Local directory to download the images")
    parser.add_argument("--output_gif", type=str, required=True, help="Path to save the generated GIF (e.g., './output.gif')")
    
    args = parser.parse_args()
    
    generate_gif_from_wandb(args.run_path, args.download_path, args.output_gif)
