import os

def remove_files_from_dir(folder_path, file_start, file_extension):
    """
    Semantics:
        Remove files from a folder at folder_path.
    Args:
        folder_path(str): The path to the folder.
        file_start(str): The string the file name starts with.
        file_extension(str): The file extension.

    Returns:
        Void.
    """
    for file in os.listdir(folder_path):
        if file.startswith(file_start) and file.endswith(file_extension):
            file_path = os.path.join(folder_path, file)
            os.remove(file_path)
