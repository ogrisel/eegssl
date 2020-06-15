import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPool1D
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.layers import concatenate


def make_encoder_model(
    input_shape,
    n_blocks=3,
    base_filters=64,
    kernel_size=11,
    pool_strides=5,
    dilation_rate=2,
):
    filters = base_filters
    model = Sequential()
    for block_idx in range(n_blocks):
        model.add(
            Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                input_shape=input_shape if block_idx == 0 else (),
                dilation_rate=dilation_rate,
                activation="relu",
            )
        )
        if block_idx < n_blocks - 1:
            model.add(MaxPool1D(strides=pool_strides))
            filters *= 2

    model.add(GlobalAveragePooling1D())
    model.add(LayerNormalization())
    return model


class PairModel(tf.keras.models.Model):
    def __init__(
        self,
        input_shape,
        n_blocks=3,
        base_filters=64,
        kernel_size=11,
        pool_strides=5,
        dilation_rate=2,
        hidden_size=256,
    ):
        super().__init__()
        self.encoder = make_encoder_model(
            input_shape=input_shape,
            n_blocks=n_blocks,
            base_filters=base_filters,
            kernel_size=kernel_size,
            pool_strides=pool_strides,
            dilation_rate=dilation_rate,
        )
        self.mlp_hidden = Dense(units=hidden_size, activation="relu")
        self.mlp_output = Dense(units=1)

    def call(self, inputs, **kwargs):
        a, b = inputs
        a = self.encoder(a, **kwargs)
        b = self.encoder(b, **kwargs)
        x = concatenate([a, b, a - b, a * b], axis=-1)
        x = self.mlp_hidden(x)
        return self.mlp_output(x)
