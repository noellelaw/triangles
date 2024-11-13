import os
import pandas as pd
import colorsys

def generate_rgb_colors(num_colors):
    """
    Generate RGB colors for labels if grey scale images are provided 
    """
    colors = []
    for i in range(num_colors):
        # Use the hue value (HSL) to create distinct colors, evenly spaced
        hue = i / num_colors
        lightness = 0.5
        saturation = 0.8
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        
        # Convert RGB from 0-1 range to 0-255 and format as an RGB string
        rgb = tuple(int(c * 255) for c in rgb)
        colors.append(f"rgb{rgb}")
    return colors


def check_and_make_dirs(folder_path):
    """
    Check if a directory exists at the given folder_path.
    If it does not exist, create the directory.

    Parameters:
    folder_path (str): The path to the directory.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Experiment directory created: {folder_path}")
    else:
        print(f"Directory already exists: {folder_path}")
    
def create_output_folders(root, folder_path, label_data):
    """
    Make output directors according to uploaded zip file 

    Parameters:
    root (str): The root path
    folder_path (str): The path to the directory.
    label_data (dictionary): User-inputed label, class, and category data.
    """
    experiment = folder_path.split('/')[-1]
    output_folder = os.path.join(root, experiment)
    # Check and make the top level output folder
    check_and_make_dirs(output_folder)
    # Check and make the metadata folder to save label data
    metadata_folder = os.path.join(output_folder, 'metadata')
    check_and_make_dirs(metadata_folder)
    # Save label data when user has inputted values
    if len(label_data) > 1:
        pd.DataFrame(label_data).to_csv(os.path.join(metadata_folder, 'label_data.csv'), index=False)
    # Check and make the visualizations folder to save plots
    check_and_make_dirs(os.path.join(output_folder, 'visualizations'))
    # Check and make the metrics folder to save metrics
    check_and_make_dirs(os.path.join(output_folder, 'metrics'))
    
    return output_folder