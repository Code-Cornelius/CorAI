import json
import os

from priv_lib_util.tools.src.function_json import zip_json, unzip_json


def list_of_dicts_to_txt(parameter_options, column_size=15, file_name="config.txt"):
    """
        Writes the parameter options in a formatted file, the header of the file contains the parameter names,
        each following line contains an entry from the parameter options.
    Args:
        parameter_options: The list of dictionaries to be written to the file
        column_size: The size of the columns in the file
        file_name: The path to where the config file should be written

    Returns:
        None
    """
    # get the names of all the parameters
    p_names = list(parameter_options[0])

    # get the number of values for each parameter
    length = len(p_names)

    # start with the line name
    line_pattern = ""
    for i in range(length):
        line_pattern += " {:>" + str(column_size) + "}"
    line_pattern += "\n"

    with open(file_name, "w") as file:

        line = line_pattern.format(*p_names)
        file.write(line)

        for p_option in parameter_options:
            values = []
            for p_name in p_names:
                values.append(p_option[p_name])

            line = line_pattern.format(*values)
            file.write(line)


def list_of_dicts_to_json(parameter_options, file_name="config.json", compress=False):
    """
        Writes the parameter options to a json file.
        Create a directory if the path yields a non-existent directory.
    Args:
        parameter_options: The list of dictionaries to be written to the file
        file_name: The path to where the config file should be written
        compress: Boolean to specify if compression should be applied before writing to the file

    Returns:
        None
    """

    if compress:
        parameter_options = zip_json(parameter_options)

    directory_where_to_save = os.path.dirname(file_name)
    if not os.path.exists(directory_where_to_save):
        if directory_where_to_save != '':
            os.makedirs(directory_where_to_save)
    with open(file_name, 'w') as file:
        json.dump(parameter_options, file)

def json2python(path, compress = False):
    with open(path, 'r') as file:
        dict = json.load(file)
        if compress:
            dict = unzip_json(dict)
        file.close()
    return dict


def factory_fct_linked_path(ROOT_DIR, path_to_folder):
    # example:
    #   path_save_history = linked_path(['plots', f"best_score_{nb}"])
    # and ROOT_DIR should be imported from a script at the root where it is written:
    # import os
    # ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    PATH_TO_ROOT = os.path.join(ROOT_DIR, path_to_folder)

    def linked_path(path):
        # a list of folders like: ['C','users','name'...]
        # when adding a '' at the end like
        #       path_to_directory = linker_path_to_result_file([path, ''])
        # one adds a \ at the end of the path. This is necessary in order to continue writing the path.
        return os.path.join(PATH_TO_ROOT, *path)

    return linked_path

