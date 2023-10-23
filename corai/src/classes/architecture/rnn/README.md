When using RNNs, the easiest way is to initialise a module that stores the hidden states `h0`, and then the cells. An
example:

```python
    seq_nn = [
        corai.One_hidden_recurrent(num_layers, int(bidirectional) + 1, hidden_size),
        (model := corai.factory_parametrised_RNN(input_dim=input_size, output_dim=output_size,
                                                 num_layers=num_layers, bidirectional=bidirectional,
                                                 nb_output_consider=lookforward_window,
                                                 hidden_size=hidden_size, dropout=dropout,
                                                 rnn_class=nn.GRU)()),  # walrus operator Python 3.8
        corai.Reshape([-1, model.output_len]),
        nn.Linear(model.output_len, hidden_FC, bias=True),
        nn.CELU(),
        nn.Linear(hidden_FC, lookforward_window * output_size, bias=True),
        corai.Reshape([-1, lookforward_window, output_size]),
    ]
```
