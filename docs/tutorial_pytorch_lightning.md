# Pytorch Lightning (PL)

Pytorch lightning is a library that reduces the boilerplate of Pytorch. We consider that this project is robust and
efficient in its task, and for this reason we incorparate it in our projects. The idea is that it does some work we were
trying to do ourselves: `metrics`, `early_stoppers`...

For this reason, we stopped supporting the train functions of CorAI and instead, we recommend using Pytorch Lightning.
However, some functionalities from CorAI are still beneficial, and some important to us functionalities do not appear in
PL. For this reason, we created some PL classes exposed at the root, and created adaptors between these classes and in
particular, the `Estimator_History`.

# History_dict

A logger, including a live performance checker and is convertible to `Estimator_History`.

# progressbar_without_val_batch_update

It replaces the original progressbar, which is buggy on IDE.