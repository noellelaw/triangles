import os
import pandas as pd


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