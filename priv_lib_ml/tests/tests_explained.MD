## Tests

`simple_benchmark_cpu_gpu_size_input` is a script for testing the performances of the computer.
Fix the parameter: `READ_FROM_SAVED`


* `test_estim_history` tests that the class estim_history is ok.

* `test_fcnn` fcnn related fcts.

* `test_training` verifies that the trainings functions are correct.
There are two tasks, a classification problem and a regression problem. We verify it works as expected by verifying the error is small enough.

* Finally, in `examples_of_tasks`, one finds a bunch of example tasks.
These are not "testable" because they might require a GPU to be tested in a decent time.
The tested tasks are in `copy_of_test_examples`. 


For example, in `example_hyper_param`, one can set two parameters:

```python
NEW_DATASET = False 
SAVE_TO_FILE = True
```

#Comment about source:

good practice would be to call the src things:

`corai`

so in the future, use:

```python
import priv_lib_ml as corai
```

in particular, avoid importing everything as it imports `pytorch`. The risk being that there are some naming conflicts.