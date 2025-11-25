import gdown
import os
import shutil

def download_file(url: str, output_path: str, unzip: bool) -> None:
    """
    Downloads a file from the specified URL to the given output path.

    Args:
        url (str): The URL of the file to download.
        output_path (str): The local path where the file will be saved.
    """

    # Create the output directory if it doesn't exist
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Download the file
        gdown.download(url, output_path, quiet=False, fuzzy=True)

        # Unzip the file if required
        if unzip and output_path.endswith('.zip'):
            shutil.unpack_archive(output_path, os.path.dirname(output_path))
            print(f"📦 File unzip on path: {os.path.dirname(output_path)}") 
    else:
        print(f"⚠️  File already exists: {output_path}. Skipping download.")