from corai import Estim_history

from corai.tests.pytorch_light.example_sinus_no_comments import Sinus_model

checkpoint_path = r"/Users/biancateodoracatea/PycharmProjects/python_libraries/corai/pytorch_light/out/default_Logger_custom_plot/21_1.0/checkpoints/epoch=49-step=449.ckpt"
history_path = r"/Users/biancateodoracatea/PycharmProjects/python_libraries/corai/pytorch_light/out/estims/estim1.json"

model = Sinus_model.load_from_checkpoint(checkpoint_path)
estim = Estim_history.from_json(history_path)

