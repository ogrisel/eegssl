# %load_ext autoreload
# %autoreload 2

# %%
from time import time
import tensorflow as tf
import matplotlib.pyplot as plt  # noqa

from eegssl.data_loader import PairGenerator
from eegssl.data_loader import RawPhysionetLoader
from eegssl.model import PairModel


# %%
print("Loading data...")
physionet_loader = RawPhysionetLoader()

pair_gen_train = PairGenerator(
    physionet_loader.load(split="train"), pairs_per_chunk=100, preload=False
)
pair_gen_test = PairGenerator(
    physionet_loader.load(split="test"), pairs_per_chunk=100, preload=False
)


# %%
print("Wrapping as a TensorFlow Dataset")

with tf.device("/GPU:0"):
    ds_pairs_train = pair_gen_train.to_tf_dataset()
    ds_pairs_test = pair_gen_test.to_tf_dataset()


# %%
print("Loading a fixed validation set in GPU memory")

with tf.device("/GPU:0"):
    # Fixed validation set, preloaded in GPU memory
    validation_data = next(iter(ds_pairs_test.batch(10000)))


# %%
with tf.device("/GPU:0"):
    model = PairModel(
        input_shape=(pair_gen_train.window_size, pair_gen_train.channel_size)
    )


# %%
# Time the forward pass with a single GPU-preloaded batch of data
batch_size = 32
with tf.device("/GPU:0"):
    # Fixed batch set, preloaded in GPU memory to bench the forward pass
    one_batch_inputs, one_batch_label = next(iter(ds_pairs_test.batch(batch_size)))

_ = model(one_batch_inputs)
n_passes = 300
t0 = time()
for _ in range(n_passes):
    _ = model(one_batch_inputs)
duration = (time() - t0) / n_passes
total_bytes = sum(a.numpy().nbytes for a in one_batch_inputs)
print(
    f"Forward pass takes {duration:.3f}s " f"({total_bytes / duration / 1e6:.1f} MB/s)"
)

# %%
# Time the forward pass with on-the-fly training data loader
_ = model(one_batch_inputs)
n_batches = 300
t0 = time()
for input_batch, _ in iter(ds_pairs_train.batch(batch_size).take(n_batches)):
    _ = model(input_batch)
duration = (time() - t0) / n_batches
print(
    f"Forward pass takes {duration:.3f}s " f"({total_bytes / duration / 1e6:.1f} MB/s)"
)


# %%
steps_per_epoch = 1000
epochs = 300
lr_schedule = tf.keras.experimental.CosineDecay(
    initial_learning_rate=1e-3, decay_steps=steps_per_epoch * epochs
)
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])


# %%
loss, accuracy = model.evaluate(*validation_data)


# %%
history = model.fit(
    ds_pairs_train.batch(batch_size),
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=validation_data,
)

# %%
model.evaluate(ds_pairs_test.batch(1000).take(100))

# %%
