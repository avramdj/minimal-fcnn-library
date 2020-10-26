# Minimal FCNN Library
A minimal fully connected neural network/multilayer perceptron library from scratch in C++.
Inspired by [Keras](https://github.com/keras-team/keras).

Example

```c++
//creating sequential fully connected model
model::Net model;

model.add_layer(in_size);
model.add_layer(16);
model.add_layer(32);
model.add_layer(32);
model.add_layer(16);
model.add_layer(out_size);

float train_rate = 0.1;
float batch_size = 32;
model.compile(train_rate, batch_size);

model.fit(*train_data_in, *train_data_out, 20);

model.evaluate(*train_data_in, *train_data_out);
```

<img src="img/ss1.png" width="600px">
