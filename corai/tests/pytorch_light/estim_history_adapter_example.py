from corai import Estim_history

log_path = r"/Users/biancateodoracatea/PycharmProjects/python_libraries/corai/pytorch_light/out/csv_logs/default/version_0/metrics.csv"
ckpt = r"/Users/biancateodoracatea/PycharmProjects/python_libraries/corai/pytorch_light/out/default_default_Logger_custom_plot/2_6_1.0/checkpoints/epoch=259-step=2339.ckpt"

estimator = Estim_history.from_pl_logs(log_path=log_path, checkpoint_path=ckpt)

print(estimator)