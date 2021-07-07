import os

def clean_folder(folder_path, file_start, file_extension):
    for file in os.listdir(folder_path):
        if file.startswith(file_start) and file.endswith(file_extension):
            file_path = os.path.join(folder_path, file)
            os.remove(file_path)
