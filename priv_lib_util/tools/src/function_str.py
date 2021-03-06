def test_print(my_str):
    print("! test ! ", my_str)


def tuple_to_str(my_tuple, delimiter=''):
    """
     Semantics:
        Convert tuples to strings.

    Args:
        my_tuple: a tuple to be converted into a string
        delimiter: tuple[0] + delimiter + tuple[1] + delimiter + ...

    Returns:
        returns the converted string.

    """
    my_str = ''
    for i in range(len(my_tuple)):
        my_str += str(my_tuple[i]) + delimiter

    return my_str
