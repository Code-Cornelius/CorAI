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

    Args:
        parameter_options:

    Returns:

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
