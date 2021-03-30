from unittest import TestCase

from priv_lib_util.tools.src.function_dict import parameter_product

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
