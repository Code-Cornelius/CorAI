# Saving/Loading a net

### Savable Nets

The class `savable_net` implements the method ....


### 1. Saving a net

A good practice is to save the net with a `.pth` extension, as recommended:
(cf. https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_models_for_inference.html).

```python
net.save_net(path)
```

### 2. Loading a net
A net can be loaded using the `load_net` function.
todo explain how it works, the dict, what is net here, initialised object with untrained parameters (I think the latter)
```python
net.load_net(path)
```