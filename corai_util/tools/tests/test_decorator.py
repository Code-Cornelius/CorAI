from unittest import TestCase
import os

from corai_util.tools.src.decorator import decorator_delayed_keyboard_interrupt


class Test_decorator(TestCase):
    def test_memoization(self):
        pass

    def test_timer(self):
        pass

    def test_set_new_methods(self):
        pass

    def test_prediction_total_time(self):
        pass

    @decorator_delayed_keyboard_interrupt
    def write_experiment(self):
        file_path = "test_file.txt"

        with open(file_path, 'a'):
            os.utime(file_path, None)

        with open(file_path, 'a') as file:
            for i in range(100000):
                print(i)
                file.write(str(i))

        os.remove(file_path)

    def test_delayed_keyboard_interrupt(self):
        self.write_experiment()
