from unittest import TestCase

from priv_lib_util.tools.src.function_file import remove_files_from_dir

from nn_classes.estimator.history.estim_history import Estim_history
import numpy as np
import os

from nn_classes.estimator.hyper_parameters.estim_hyper_param import Estim_hyper_param
from nn_train.kfold_training import _translate_history_to_dataframe

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
train_history = {
    'training': {
        'loss': np.array(range(10)),
        'L1': np.array(range(10, 20)),
        'L2': np.array(range(20, 30))
    },
    'validation': {
        'loss': np.array(range(30, 40)),
        'L1': np.array(range(40, 50)),
        'L2': np.array(range(50, 60))
    },
    'best_epoch': 2
}
metric_names = ['L1', 'L2']

flattened_history = {
    'epoch': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    'loss_training': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    'L1_training': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    'L2_training': [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
    'loss_validation': [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
    'L1_validation': [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
    'L2_validation': [50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
    'fold': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}
FOLDER_PATH = os.path.join(ROOT_DIR, "tmp_param_train_test")

class Test_estim_history(TestCase):

    def test_create_estim_with_validation_has_correct_column_names(self):
        estimator = Estim_history(metric_names=metric_names, validation=True)

        column_names = ['fold', 'epoch', 'loss_training', 'L2_training', 'L3_training',
                        'loss_validation', 'L2_validation', 'L3_validation'].sort()

        df_column_names = estimator._df.columns.values.sort()

        assert column_names == df_column_names

    def test_create_estim_without_validation_has_correct_column_names(self):
        estimator = Estim_history(metric_names=metric_names, validation=False)

        column_names = ['fold', 'epoch', 'loss_training', 'L2_training', 'L3_training'].sort()

        df_column_names = estimator._df.columns.values.sort()

        assert column_names == df_column_names

    def test_translate_flattens_the_history(self):
        translated_history = _translate_history_to_dataframe(train_history, 1, True)

        assert translated_history == flattened_history

    def test_append_history_from_folds_to_estim(self):
        estimator = Estim_history(metric_names=metric_names, validation=True)

        history = _translate_history_to_dataframe(train_history, 0, True)
        estimator.append(history, fold_best_epoch =  2, fold_time = 3)

        history = _translate_history_to_dataframe(train_history, 1, True)
        estimator.append(history, fold_best_epoch = 3, fold_time = 3)

        df = estimator._df

        assert df.shape == (20, 8)


    def test_to_csv(self):
        file_name = "test.json"

        path = os.path.join(FOLDER_PATH, file_name)
        estimator = Estim_history(metric_names=metric_names, validation=True)

        history = _translate_history_to_dataframe(train_history, 0, True)
        estimator.append(history, 2, 2)

        history = _translate_history_to_dataframe(train_history, 1, True)
        estimator.append(history, 3, 2)

        estimator.to_json(path)

        new_estim = Estim_history.from_json(path)

        print(new_estim)
        remove_files_from_dir(folder_path=FOLDER_PATH, file_start="", file_extension="")

    def test_test_Test(self):
        estimator = Estim_history(metric_names=metric_names, validation=True,
                                  hyper_params={'optim': 'adam',
                                                       'lr': 0.7,
                                                       'layers': [12, 10]})

        history = _translate_history_to_dataframe(train_history, 0, True)
        estimator.append(history, fold_best_epoch= 2, fold_time= 3)
        estimator.best_fold = 0 # manually set, normally set during training.

        file_path1 = os.path.join(FOLDER_PATH, "estim_1.json")
        estimator.to_json(file_path1)

        history = _translate_history_to_dataframe(train_history, 1, True)
        estimator.append(history, 3, 2)

        estimator.best_fold = 1
        estimator.slice_best_fold()

        file_path2 = os.path.join(FOLDER_PATH, "estim_2.json")
        estimator.to_json(file_path2)

        estimator_param = Estim_hyper_param.from_folder(FOLDER_PATH, metric_names=["loss_validation"])

        print("original estimator : \n", estimator)
        print("hyper param estimator : \n", estimator_param)
        remove_files_from_dir(folder_path=FOLDER_PATH, file_start="", file_extension="")