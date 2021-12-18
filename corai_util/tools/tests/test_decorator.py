from unittest import TestCase
import os

class Test_decorator(TestCase):
    def test_memoization(self):
        pass

    def test_timer(self):
        pass

    def test_set_new_methods(self):
        pass

    def test_prediction_total_time(self):
        pass

    def test_delayed_keyboard_interrupt(self):
        file_path = "test_file.txt"

        with open(file_path, 'a'):
            os.utime(file_path, None)

        with open(file_path, 'a') as file:
            for i in range(1000000):
                file.write(str(i))

        os.remove(file_path)