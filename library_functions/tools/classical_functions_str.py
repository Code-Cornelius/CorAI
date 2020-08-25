def test_print(my_str):
    print("! test ! ", my_str)


def tuple_to_str(my_tuple):
    my_str = ''
    for i in range(len(my_tuple)):
        my_str += str(my_tuple[i]) + '_'

    return my_str
