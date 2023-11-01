import os
import shutil
import glob

def copy_jpeg_files(input_directory, output_directory):
    for root, _, files in os.walk(input_directory):
        for file in files:
            if file.lower().endswith('.jpeg'):
                source_path = os.path.join(root, file)
                subfolder_name = os.path.basename(root)
                file_parts = file.split('.')
                file_without_extension = file_parts[0]
                new_filename = f"{subfolder_name}_{file_without_extension}.jpeg"
                destination_path = os.path.join(output_directory, new_filename)
                shutil.copy(source_path, destination_path)
                print(f"Copied '{source_path}' to '{destination_path}'")

input_directory = "../NF1 Extract October/"
output_directory = "./nf1_database/october/"

copy_jpeg_files(input_directory, output_directory)
