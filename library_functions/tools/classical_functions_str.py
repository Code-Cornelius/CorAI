def test_print(my_str):
    print("! test ! ", my_str)


def tuple_to_str(my_tuple, delimiter=''):
    """ convert tuples to strings.

    Args:
        my_tuple: a tuple I want to convert into a string
        delimiter: tuple[0] + delimiter + tuple[1] + delimiter + ...

    Returns:

    """
    my_str = ''
    for i in range(len(my_tuple)):
        my_str += str(my_tuple[i]) + delimiter

    return my_str
