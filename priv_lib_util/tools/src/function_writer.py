import json


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

        file.close()


def list_of_dicts_to_json(parameter_options, file_name="config.json"):
    """
        Writes the parameter options to a json file
    Args:
        parameter_options: The list of dictionaries to be written to the file
        file_name: The path to where the config file should be written

    Returns:
        None
    """
    with open(file_name, 'w') as file:
        json.dump(parameter_options, file)