import json

def up(my_dict, new_dict):
    """
    Semantics:
        If a key from new_dict is not defined in my_dict, add it.
        The behaviour is almost like the in-built function update.

    Args:
        my_dict: old to update.
        new_dict: the new entries

    Returns:
        void, the dict is updated since the dict is mutable.

    """
    for key in new_dict:
        if key not in my_dict:
            my_dict[key] = new_dict[key]

    return


def parameter_product(parameter_options):
    """
    Function to compute the product between lists of parameters
    Args:
        parameter_options: a dictionary composed of possible parameters. Each parameter should have a list of
            the values said parameter can take

    Returns:
        A list of dictionaries. Each dictionary will contain a unique combination of the parameters.
    """

    result = []
    p_names = list(parameter_options)

    def product(p_names, options_dict, i=0):
        """backtracking."""

        if i == len(p_names):
            copy = dict(options_dict)
            result.append(copy)
            return

        for option in parameter_options[p_names[i]]:
            options_dict[p_names[i]] = option
            product(p_names, options_dict, i+1)

    product(p_names, {})
    return result


def replace_function_names_to_functions(parameters_at_index, mapping):
    """
        Translates parameters from string to the corresponding function

    Args:
        parameters_at_index: dictionary with potential parameters to be translated to the corresponding function
        mapping: dictionary with a string key corresponding to the name of the function and the corresponding function
            as a value
    Return
        None
    """

    for param, value in parameters_at_index.items():
        # only string values will be replaced
        if isinstance(value, str):

            # if the value has a mapping, change the value to the mapping
            if value in mapping:
                parameters_at_index[param] = mapping[value]


def retrieve_parameters_by_index_from_json(index, file_path):
    """
        Retrieve the dictionary at the specified index from a list stored in a json file

    Args:
        index: index of the dictionary to be returned
        file_path: the path to the json file where the information should be read from

    Returns:
        dictionary at index from json file
    """
    with open(file_path, 'r') as file:
        parameters = json.load(file)
        file.close()

    assert 0 <= index < len(parameters), 'Parameter index is outside the bounds'

    return parameters[index]