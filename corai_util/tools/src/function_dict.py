import json

from corai_util.tools.src.function_iterable import is_iterable


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


def replace_function_names_to_functions(parameters_at_index, mapping, silent=True):
    """
        Translates parameters from string to the corresponding function

    Args:
        parameters_at_index: dictionary with potential parameters to be translated to the corresponding function
        mapping: dictionary with a string key corresponding to the name of the function and the corresponding function
            as a value
        silent: debug
    Return
        None
    """

    for param, value in parameters_at_index.items():
        # case string values
        if isinstance(value, str):

            # if the value has a mapping, change the value to the mapping
            if value in mapping:
                parameters_at_index[param] = mapping[value]

        # case list of string values
        if isinstance(value, list):

            if not silent:
                print(f"In the list: {param} the following values were not mapped to a function:")

            for i, func in enumerate(value):

                # check if the value is in the mapping
                if isinstance(func, str) and func in mapping:
                    parameters_at_index[param][i] = mapping[func]
                elif not silent:
                    print(f"        -> {func}")


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

    assert 0 <= index < len(parameters), "Parameter index is outside the bounds (number of settings)."

    return parameters[index]

def filter(names, keys, filter_rules):
    if filter_rules is None:
        return keys
    unwanted = []
    for key in keys:
        for i, name in enumerate(names):
            if name in filter_rules:
                # here we distinguish the case simple or multi keys. If multi keys, key is a tuple.
                if is_iterable(key):
                    if key[i] in filter_rules[name]:
                        unwanted.append(key)
                else:
                    if key in filter_rules[name]:
                        unwanted.append(key)

    return [key for key in keys if key in unwanted]


"""
example:

parameterstest = {'SEED': [42],
                  'HIDDEN_SIZES': [[16, 16, 32, 32, 16, 8],
                                   [8, 16, 8, 4]],
                  'ACTIVATION_FUNCTIONS': [['Tanh', 'Celu', 'Tanh', 'Celu', 'Celu', 'Celu']],
                  'BATCH_SIZE': [0],
                  'DROPOUT': [0.],
                  'OPTIMISER': ['Adam'],
                  'DICT_OPTIMISER': [{'lr': 0.001, 'weight_decay': 1E-8}]
                  }

if __name__ == '__main__':
    # convert them to the product
    product_param = function_dict.parameter_product(parametersTASK3_1)
    # write them down with the given name
    path = "Task3/data/parameters_grid_search.json"
    function_writer.list_of_dicts_to_json(product_param, file_name=path)
    print(f"File {path} has been updated.")
    print(f"    Number of configurations: {len(product_param)}.")



"""