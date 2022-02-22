Hyperparameter selection is extremely important for deployment. One trains the model for a variety of hyperparameters
and see which one gives the best result. The question that remains, how to do this efficiently?

# Grid Search

We develop an efficient tool when one uses a grid search. An example lies for CorAI pipeline
in `tests\examples_of_tasks\example_hyper_param.py` and with PL in `tests\pytorch_light\example_hyper_param.py`. The
object used for comparison is the `Estim_hyper_param ` which can be instantiated by giving a path to a folder full of
estimators history where the data lies. In the following, we distinguish multiple cases how one could want to create
estimators.

In order to see the global idea of how to iterate over different models and other tricks, please refer
to   `corai/src/classes/estimator/history/Tutorial_estim_hyperparameter.md`. In this tutorial, we mainly focus on
alternative ways to create `Estim_history` and `Estim_hyper_param`.

## From nn_kfold

The easiest way to create estimators is with

```python
import corai
from corai_util.tools import function_writer

linker_estims = function_writer.factory_fct_linked_path(ROOTPATH, "histories_path")
linker_models = function_writer.factory_fct_linked_path(ROOTPATH, "models_path")
(net, estimator_history) = corai.nn_kfold_train(train_X, train_Y,
                                                Class_Parametrized_NN,
                                                param_train=param_training,
                                                early_stoppers=early_stoppers,
                                                nb_split=1, shuffle_kfold=True,
                                                percent_val_for_1_fold=20,
                                                silent=True, hyper_param=params)
estimator_history.to_json(path=linker_estims([f"estim_{i}.json"]), compress=False)
net.save_net(path=linker_models([f"model_{i}.pth"]))
```

## From Pytorch Lightning

```python
import corai
from corai_util.tools import function_writer

linker_estims = function_writer.factory_fct_linked_path(ROOTPATH, "histories_path")
linker_models = function_writer.factory_fct_linked_path(ROOTPATH, "models_path")
start_time = time.perf_counter()
trainer.fit(sinus_model, datamodule=sinus_data)
final_time = time.perf_counter() - start_time
train_time = np.round(time.perf_counter() - start_time, 2)
print("Total time training: ", train_time, " seconds. In average, it took: ",
      np.round(train_time / trainer.current_epoch, 4), " seconds per epochs.")
estimator_history = logger_custom.to_estim_history(checkpoint=chckpnt, train_time=final_time)
estimator_history.to_json(linker_estims([f'estim_{i}.json']), compress=False)
# todo the model path
estims.append(estimator_history)
```

## When one has just a model: from scratch

```python
```

# Conclusion

A Hyper-parameter tuning script should contain three parts:

1. The arguments iterated over,
2. A function that converts the parameter from the iteration into a usable architecture, which in turn could require a
   mapping from string to functions,
3. the generator of estim_history, which iterates over the different architecture, in order to create and save
   the `estim_history`.