
def write_config_file_for_parameters(parameter_options, column_size=15, file_name="config.txt"):

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

