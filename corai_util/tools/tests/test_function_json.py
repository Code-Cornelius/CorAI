from unittest import TestCase

from corai_util.tools.src.function_json import zip_json, unzip_json

import json, os

from corai_util.tools.src.function_writer import list_of_dicts_to_json


class Test_function_json(TestCase):

    def test_zip_and_unzip_return_the_right_result(self):

        unzipped_json = {
            'a': [1, 2, 3],
            'b': ['hello', 'hi'],
            'c': 27
        }

        zipped_json = {'base64(zip(o))': 'eJyrVkpUslKINtRRMNJRMI7VUVBKAvGVMlJzcvKVgNyMTCWQaDJQ1Mi8FgAKsgtR'}

        assert zip_json(unzipped_json) == zipped_json
        assert unzipped_json == unzip_json(zipped_json)
        assert unzipped_json == unzip_json(zip_json(unzipped_json))
        assert zipped_json == zip_json(unzip_json(zipped_json))

    def test_compression_rate(self):
        # replace with file to check
        input_name = "some_file"

        with open(f'generated_test_files/{input_name}.json', 'r') as file:
            parameters = json.load(file)

        list_of_dicts_to_json(parameters, f'generated_test_files/compressed_{input_name}.json', True)

        original_size = os.path.getsize(f'generated_test_files/{input_name}.json')
        compressed_size = os.path.getsize(f'generated_test_files/compressed_{input_name}.json')

        print(f"File compressed from {original_size} byte to {compressed_size} bytes")
        print(f"new file is {(1 - compressed_size/original_size) * 100}% smaller")