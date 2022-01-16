# Ideas for continuing the project:


* **Estimator**
    * for hawkes estim, not mandatory to have true values. in init, in the estim plotters.
    * relplot, in lineplot write dependencies.
      
    * scatter: write with ax and save
    * a function on line 345 to write, there is an example of how to do it.
    * function applied to a column (and create a new column)
  
    * change name of super draw to something else
    * make readme tutorial about the whole pipeline training + selecting best model + prediction with the best model
      


* **APLOT**. The 3D plot have a different logic, due to the fact that axis is 3D axis. How to deal with it ? 3DAPlot?

  perhaps an idea, split APlot into: displayable plot -> grid_plot -> APlot.

  the grid_plot would take care of all the parameters, and in particular, would handle the dict_plot_param, dict_ax...
  the issue with some parameters is that for dict_plot_param they are just given to the plotter. It should be filtered
  first. This is the reason to problems in 3D axis.

  For example, the cmap is a pain. grid_plot could create a function that does this, everything a function needs to
  check before starting the inside:
  then one just needs to wrap all the function inside APlot. that is actually my main struggle: how to deal with
  function that can take any keyword argument and has a behavior depending on that? like colors, linewidth etc...

  - change the nb_ax by index_ax.
  - homogeneous input, not nb_ax then xx but xx then nb_ax. It should always be the same order. A possibility would be
    to put first the data, then the axis.
  - change return to give an ax. such that one can continue drawing on an axis!
  - what is happening with bis axis is a bit obscure. Let s clarify it. Not sure how.
  - labels when both axis are on the same graph, see if there is any comment about it and change it to adapt the new
    behavior.
  - verify that if I access a nb_ax, I also check that the number is correct!
  - put the test at the bottom into a right test.
  - when dict of parameters given, mention if some are unused.



*Verify if it is L2 error or L2 squared. Indeed, we should say MSE and not L2.
* asserts when reading loading json for estimator history to avoid obscure error message.
* dtype for the architectures.




Work January CorAI 1.400:

- remove the two types of RNN, and only have one type which take in input the hidden states.

- using history without training: there is an example, it shows that CorAI not adapted. Construct adaptors to not have
  to change fundamentally the code.

- for hyperparam tuning:
  - make an example, where I do not use anything complicated.
    - I need two examples, one using pl,
    - one using no training and building from scratch estimator. The logger could be use, and manually compute losses.
  - you can use the two previous examples.

- update tutorial collab,
- examples for loading things in pl checkpoints which are Bianca path dependent.

- RNN parameters are not used. Clean the file.