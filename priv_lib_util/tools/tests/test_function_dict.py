import os
from collections import Callable
from unittest import TestCase

from config import ROOT_DIR
from priv_lib_util.tools.src.function_dict import parameter_product, replace_function_names_to_functions, \
    retrieve_parameters_by_index_from_json
from priv_lib_util.tools.src.function_writer import list_of_dicts_to_json

PATH = os.path.join(ROOT_DIR, 'priv_lib_util', 'tools', 'tests', 'generated_test_files')

test_input = {
    'parameter_1': [1, 2],
    'parameter_2': ['a', 'b']
}

test_output = [{
    'parameter_1': 1,
    'parameter_2': 'a'
    }, {
    'parameter_1': 1,
    'parameter_2': 'b'
    }, {
    'parameter_1': 2,
    'parameter_2': 'a'
    }, {
    'parameter_1': 2,
    'parameter_2': 'b'
    },
]

function_test_dict = {
    'a_func': 'len',
    'a_number': 12,
    'a_list': [1, 2, 3],
    'a_string': 'aaa'
}

function_map = {
    'len': len,
    'sum': sum
}

class Test_function_dict(TestCase):
    def test_up(self):
        pass

    def test_parameter_product_returns_right_number_of_elements(self):
        result = parameter_product(test_input)
        length = 1
        for parameter_name in test_input:
            length *= len(test_input[parameter_name])
        assert len(result) == length

    def test_parameter_product_returns_the_product_of_parameter_options(self):
        result = parameter_product(test_input)

        assert result == test_output

    def test_replace_function_names_to_functions_doesnt_modify_unnecessary_params(self):
        replace_function_names_to_functions(function_test_dict, function_map)

        assert isinstance(function_test_dict['a_number'], int)
        assert isinstance(function_test_dict['a_list'], list)
        assert isinstance(function_test_dict['a_list'][0], int)
        assert isinstance(function_test_dict['a_string'], str)

    def test_replace_function_names_to_functions_replaces_with_function(self):
        replace_function_names_to_functions(function_test_dict, function_map)

        len_func = function_test_dict['a_func']

        assert len_func(function_test_dict['a_list']) == 3


    def test_retrieve_parameters_by_index_from_json_return_correct_dictionary(self):
        file_name = "test_output.json"
        file_path = os.path.join(PATH, file_name)

        dict_one = retrieve_parameters_by_index_from_json(0, file_path)
        assert dict_one == test_output[0]

        dict_two = retrieve_parameters_by_index_from_json(1, file_path)
        assert dict_two == test_output[1]

        dict_three = retrieve_parameters_by_index_from_json(2, file_path)
        assert dict_three == test_output[2]

        dict_four = retrieve_parameters_by_index_from_json(3, file_path)
        assert dict_four == test_output[3]
