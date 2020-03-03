# TensorFlow 2 Gradient Checkpointing
This is a simple decorator to enable gradient checkpointing (e.g. [Chen et al. (2016)](https://arxiv.org/pdf/1604.06174.pdf)) in TF2. It isn't very polished, but it's been letting me train bigger GPT-2 models on smaller hardware, so I thought I'd share it.


## Basic Usage
Use the `checkpointable` decorator to allow a function (or callable object such as a Keras `Layer`) to use gradient checkpointing. If checkpointing is desired, call the decorated function with the `_checkpoint` keyword argument set to `True`.

The example below shows a model with 40000 "layers", but checkpointing allows just 400 to be in memory at any point. On a GTX 1070 Ti, this code will result in an OOM error when the `_checkpoint` argument is set to `False`.

```python
import tensorflow as tf

from checkpointing import checkpointable

@checkpointable
def f(x, y, some_str, some_bool, z=None):
    for _ in range(200):
        x += y * z
    return x

initial = tf.ones(100000, dtype=tf.float32)
y = tf.ones(100000, dtype=tf.float32) + 1e-7
z = tf.ones(100000, dtype=tf.float32) + 1e-7
with tf.GradientTape() as g:
    g.watch(initial)
    x = initial
    for _ in range(200):
        x = f(x, y, 'a', True, z=z, _checkpoint=True)
    loss = tf.reduce_sum(x)
print(g.gradient(loss, x))
```
Arguments which are not float32 tensors (or nested list/tuple structures of such tensors) are allowed, but ignored for the purposes of gradient computation.

## Variables
If the decorated function uses variables which are not arguments, pass a list of them via the `_watch_vars` keyword argument as shown below.

```python
layer = SomeKerasLayer()
wrapped_layer = checkpointable(layer)

with tf.GradientTape() as g:
    g.watch(layer.trainable_variables)
    output = wrapped_layer(*args, **kwargs, _checkpoint=True, _watch_vars=layer.trainable_variables)
print(g.gradient(output, layer.trainable_variables))
```

## Warning: Dropout
Because gradient checkpoint relies on re-running the forward pass, stochastic layers such as a dropout will give different results for each pass. There is a hacky workaround available, which you can enable by passing `_force_seed=True` to the decorated function. This will use python's `random` library to get a random number, and set that as TensorFlow's random seed before each forward pass. If you have a better idea for addressing this issue, please do let me know.
