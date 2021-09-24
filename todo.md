# Ideas for continuing the project:

# Todos:

* **Estimator**
    * for hawkes estim, not mandatory to have true values. in init, in the estim plotters.
    * relplot, in lineplot write dependencies.
      
    * scatter: write with ax and save
    * a function on line 345 to write, there is an example of how to do it.
    * function applied to a column (and create a new column)
  
    * change name of super draw to something else
    * make readme tutorial about the whole pipeline training + selecting best model + prediction with the best model
      
# List of tasks to tackle:

* plot relplot hawkes mseerrors todo for unique MSE ( Niels will do) in draw.
* organise the init such that imports are intuitive.


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
  - homogeneous input, not nb_ax then xx then xx then nb_ax. It should always be the same order. A possibility would be
    to put first the data, then the axis.
  - change return to give an ax. such that one can continue drawing on an axis!
  - what is happening with bis axis is a bit obscure. Let s clarify it. Not sure how.
  - labels when both axis are on the same graph, see if there is any comment about it and change it to adapt the new
    behavior.
  - verify that if I access a nb_ax, I also check that the number is correct!
  - put the test at the bottom into a right test.
  - when dict of parameters given, mention if some are unused.

# More todos
- Tutorial training
  - [x] explain why one should not use the model directly to predict and what they should use instead, 
    go through each function and explain what they add 
  - [ ] add plotting explanation: regression_sin
  
- Estim_history
  - [ ] md, how to use it outside of normal pipeline
  
- Estim_hyper_param
  - [ ] md, how to use it outside of normal pipeline
  
- Hyper param tutorial
  - [x] explain what params is (in the begining)
  - explain what hp_estim is
    - [x] what do the col and line represent
    - [x] what is should contain (look at step 2)
    - [ ] what one can do with it
  - [ ] add plotting explanation: example_hyper_param  

- General
  - [ ] link other tutorials on each tutorial (ex. tutorial for estimators and plotters linked in training)
  
- [ ] Add md for packaging 
- [ ] Add md with list of all tutorials