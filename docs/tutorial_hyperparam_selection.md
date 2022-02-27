Hyperparameter selection is extremely important for deployment. One can train the model for a variety of hyperparameters
and see which one gives the best result. The question that remains, how to do this efficiently?

# Grid Search

We develop an efficient tool when one uses a grid search. An example lies for CorAI pipeline
in [`corai/tests/examples_of_tasks/example_hyper_param.py`](https://github.com/Code-Cornelius/CorAI/blob/master/corai/tests/examples_of_tasks/example_hyper_param.py)
and with PL in [`corai/tests/pytorch_light/example_hyper_param.py`](https://github.com/Code-Cornelius/CorAI/blob/master/corai/tests/pytorch_light/example_hyper_param.py).
The object used for comparison is the `Estim_hyper_param ` which can be instantiated by giving a path to a folder full of
estimators history where the data lies. In the following, we distinguish multiple cases how one could want to create
estimators.

In order to see the global idea of how to iterate over different models and other tricks, please refer
to   [`corai/src/classes/estimator/hyper_parameters/tutorial_estim_hyperparameter.md`](https://github.com/Code-Cornelius/CorAI/blob/master/corai/src/classes/estimator/hyper_parameters/tutorial_estim_hyperparameter.md). 
In this tutorial, we mainly focus on alternative ways to create `Estim_history` and `Estim_hyper_param`.

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

trainer.fit(sinus_model, datamodule=sinus_data)

estimator_history = logger_custom.to_estim_history(checkpoint=chckpnt, train_time=final_time)
estimator_history.to_json(linker_estims([f'estim_{i}.json']), compress=False)
```

In this case scenario, notice that you need to give to the method `to_estim_history` a checkpoint. The checkpoint is
used to store the parameters. The hyperparameters are stored inside the estimator and inside the checkpoint:
be careful to save all the parameters that are needed in the class
(without ignored parameters: `self.save_hyperparameters()`)
otherwise it will not be possible to reconstruct the model.

## When one has just a model: from scratch

```python
def generate_estims_history(hyper_params, input_train, output_train, input_test, output_test,
                            config_architecture, linker_estims, linker_models,
                            metrics, silent):
   estims = []
   nb_epochs = 1
   nb_prediction = 1, 1  # train , val
   for i, params in enumerate(tqdm(hyper_params)):
      current_model = config_architecture(params, input_train.shape[2], output_train.shape[2])

      corai.set_seeds(params["seed"])  # set seed for pytorch.
      ########################################################### training
      start_train_fold_time = time.time()
      _ = current_model(input_train, output_train)
      end_train_fold_time = time.time() - start_train_fold_time

      ########################################################### saving the results
      errL2_train, errL2_val = L2loss(current_model, input_train, output_train), L2loss(current_model, input_test,
                                                                                        output_test)
      errL4_train, errL4_val = L4loss(current_model, input_train, output_train), L4loss(current_model, input_test,
                                                                                        output_test)
      dict_data = {'L2_training': [errL2_train.item()], 'L2_validation': [errL2_val.item()],
                   'L4_validation': [errL4_val.item()], 'L4_training': [errL4_train.item()], 'epoch': [nb_epochs]}

      df = pd.DataFrame(dict_data)
      df['fold'] = 0  # estimators require a fold column.

      # we rename the columns so they suit the standard of estimators.
      metric_names, columns, validation = Estim_history.deconstruct_column_names(df.columns)
      df.columns = columns
      estimator = Estim_history(df=df, metric_names=[metric.name for metric in metrics], validation=validation,
                                hyper_params=params)
      estimator.list_best_epoch.append(1)
      estimator.list_train_times.append(end_train_fold_time)
      estimator.best_fold = 0
      estims.append(estimator)  # add to the list of estimators
      ########################################################### saving
      estimator.to_json(path=linker_estims([f"estim_{i}.json"]), compress=False)
      current_model.save_net(path=linker_models([f"model_{i}.pth"]))
   return estims
```

# Conclusion

A Hyper-parameter tuning script should contain three parts:

1. The arguments iterated over,
2. A function that converts the parameter from the iteration into a usable architecture, which in turn could require a
   mapping from string to functions,
3. the generator of estim_history, which iterates over the different architecture, in order to create and save
   the `estim_history`.