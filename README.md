# ConvGRU2D
Implementation of ConvGRU2D layer in tensorflow keras. 

Example:
```python
steps = 10
height = 32
width = 32
input_channels = 3
output_channels = 6

inputs = tf.keras.Input(shape=(steps, height, width, input_channels))
layer = ConvGRU.ConvGRU2D(filters=output_channels, kernel_size=3)
outputs = layer(inputs)


model = tf.keras.Model(inputs=inputs, outputs=outputs, name="convgru_model")
model.summary()
```
