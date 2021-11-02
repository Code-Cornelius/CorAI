def broken_function_with_message(error):
    a_message = ""
    try:
        raise error("A message ?")

    except error as err:
        a_message += str(err)
    finally:
        a_message += "All good"
    a_message += "I continue running !"
    return a_message


def broken_function_without_message(error):
    a_message = ""
    try:
        raise error()

    except error as err:
        a_message += str(err)
    finally:
        a_message += "All good"
    a_message += "I continue running !"
    return a_message
