# the corresponding parts in fit that are related to early stoppers are:

"""

###### ################

# Early Stopping   #

###### ################

# check if any improvement and update early.has_improved_last_epoch.

if _do_early_stop(net, early_stoppers, history, epoch, silent):

# does not break first epoch_iter because

# _early_stopped = False,

# has_improved_last_epoch = True

break # stop epoch

    # Check if NN has not improved with respect to the early stoppers.
    # If has not, we do not improve the best_weights of the NN
    if all(early_stopper.has_improved_last_epoch for early_stopper in early_stoppers):
        net.update_best_weights(epoch)
    ##################### early stop end

# ~~~~~~~~ end of the for in epoch. Training

return _return_the_stop(net, epoch, early_stoppers)

def _return_the_stop(net, current_epoch, early_stoppers):

# args should be early_stoppers (or none if not defined)

# multiple early_stoppers can't break at the same time,

# because there will be a first that breaks out the loop first.

# if no early_stopper broke, return the current epoch.

for stopper in early_stoppers:
if stopper.is_stopped():  #: check if the stopper is none or actually of type early stop.
(net.load_state_dict(net.best_weights))  # .to(device)
return net.best_epoch return current_epoch
"""