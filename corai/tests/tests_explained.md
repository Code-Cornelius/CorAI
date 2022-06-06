# Tests

`simple_benchmark_cpu_gpu_size_input.py` is a script for testing the performances of the computer. Choose the
parameter: `READ_FROM_SAVED` depending on if you want to load previous data.

## Automatic_tests

* `automatic_tests/test_estim_history.py` tests that the class `estim_history` works appropriately. We save in '
  automatic_tests/tmp_param_train_test' the temporary files.

* `automatic_tests/test_fcnn.py` fcnn related fcts.

* `automatic_tests/test_training.py` verifies that the trainings functions are correct. There are two tasks, a
  classification problem and a regression problem. We verify it works as expected by verifying the error is small
  enough.

## Examples_of_tasks

In `examples_of_tasks`, one finds a bunch of example tasks. These are not "testable" because they might require a GPU to
be tested in a decent time.

We had tested some basic examples, and we put copies of these scripts in `examples_of_tasks/copy_of_test_examples`.

In the rest of the folder, we have a few examples:

* `forecast_flight.py`,
* `testing_forecast_temperature_cnn.py`,
* `testing_forecast_temperature_rnn.py`,
* `example_hyper_param.py`,
* `colab_hyper_param.py`,

other_csv_from_examples example_hyper_param_sin_estim_models example_hyper_param_sin_estim_history param_train_test

## mnist_dataset

Folder with the data for tasks....

# Pytorch_light examples

------------------
For example, in `example_hyper_param`, one can set two parameters:

```python
NEW_DATASET = False
SAVE_TO_FILE = True
```
