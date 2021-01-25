def up(my_dict, new_dict):
    """
    SEMANTICS:
        If a key from new_dict is not defined in my_dict, add it.
        The behaviour is almost like the in-built function update.

    Args:
        my_dict: old to update.
        new_dict: the new entries

    Returns: void, the dict is updated since the dict is mutable.

    """
    for key in new_dict:
        if key not in my_dict:
            my_dict[key] = new_dict[key]

    return
