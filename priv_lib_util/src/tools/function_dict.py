def up(my_dict, new_dict):
    """
    SEMANTICS : If a key from new_dict is not defined in my_dict, add it. The behaviour is almost like update.

    Args:
        my_dict: old to update
        new_dict: the new information

    Returns: nothing, the dict is updated ! The dict is mutable.

    """
    for key in new_dict:
        if key not in my_dict:
            my_dict[key] = new_dict[key]

    return
