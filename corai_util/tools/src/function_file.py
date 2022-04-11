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


def remove_file(file_path):
    """
    Semantics:
        Wrapper around os to remove a file. It will call remove only if they file exists, nothing otherwise.

    Args:
        file_path: The full path to the file.

    Returns:
        Void.
    """
    if os.path.exists(file_path):
        os.remove(file_path)

def is_empty_file(path):
    """
    Semantics:
        Check whether the file is empty. Checks the size of it and of its subdirectory.
        Raise OSError if the file does not exist or is inaccessible.
    Args:
        path(str): The path to the file.

    Returns:
        True if the file is empty.
        False otherwise.

    References:
        https://stackoverflow.com/questions/1392413/calculating-a-directorys-size-using-python
    """
    # Using os.path.getsize() will only get you the size of the directory, NOT of its content.
    size = sum(os.path.getsize(f) for f in os.listdir('.') if os.path.isfile(f))
    return size == 0