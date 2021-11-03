import filecmp
import os
from unittest import TestCase

from config import ROOT_DIR
from corai_util.tools.src.function_writer import list_of_dicts_to_json

PATH = os.path.join(ROOT_DIR, 'corai_util', 'tools', 'tests', 'generated_test_files')

example_data = [
    {"param1": 1,
     "param2": [1, 2, 3],
     "param3": "a"},
    {"param1": 2,
     "param2": [3, 4, 5],
     "param3": "baaa"},
    {"param1": 200,
     "param2": [1, 2, 3, 7, 8],
     "param3": "some test text"}
]


class Test_function_writter(TestCase):

    def tearDown(self):
        for file in os.listdir(PATH):
            if file.startswith("test"):
                file_path = os.path.join(PATH, file)
                os.remove(file_path)

    def test_list_of_dicts_to_json_writes_the_right_info(self):
        file_name = 'json_config.json'
        file_path = os.path.join(PATH, file_name)
        test_file_path = os.path.join(PATH, f"test_{file_name}")
        list_of_dicts_to_json(example_data, test_file_path)
        assert filecmp.cmp(file_path, test_file_path, shallow=False)

