# %load_ext autoreload
# %autoreload 2

# %%
from time import time
import tensorflow as tf
import matplotlib.pyplot as plt  # noqa
import mne
from mne.datasets.sleep_physionet.age import fetch_data

from eegssl.data_loader import PairGenerator
from eegssl.model import PairModel


def fetch_raw_physionet_data(verbose=None):
    sleep_physionet_train = fetch_data(subjects=range(15), verbose=verbose)
    sleep_physionet_test = fetch_data(subjects=range(15, 20), verbose=verbose)

    mapping = {
        "EOG horizontal": "eog",
        "Resp oro-nasal": "misc",
        "EMG submental": "misc",
        "Temp rectal": "misc",
        "Event marker": "misc",
    }
    raw_train, raw_test = [], []
    for raw_data, filenames in [
        (raw_train, sleep_physionet_train),
        (raw_test, sleep_physionet_test),
    ]:
        for psg_file, hypnogram_file in filenames:
            raw_edf = mne.io.read_raw_edf(psg_file)
            raw_edf.set_annotations(mne.read_annotations(hypnogram_file))
            raw_edf.set_channel_types(mapping)
            raw_data.append(raw_edf)
    return raw_train, raw_test


raw_train, raw_test = fetch_raw_physionet_data(verbose=0)


# %%
print("Loading data...")
pair_gen_train = PairGenerator(raw_train, pairs_per_chunk=1000, preload=False)
pair_gen_test = PairGenerator(raw_test, pairs_per_chunk=1000, preload=False)


# %%
print("Wrapping as a TensorFlow Dataset")

with tf.device("/GPU:0"):
    ds_pairs_train = pair_gen_train.to_tf_dataset().prefetch(
        pair_gen_train.pairs_per_chunk * 2
    )
    ds_pairs_test = pair_gen_test.to_tf_dataset().prefetch(
        pair_gen_test.pairs_per_chunk * 2
    )


# %%
print("Loading a fixed validation set in GPU memory")

with tf.device("/GPU:0"):
    # Fixed validation set, preloaded in GPU memory
    validation_data = next(
        iter(
            ds_pairs_test.take(5 * pair_gen_test.pairs_per_chunk)
            .shuffle(buffer_size=int(1e5))
            .batch(10000)
        )
    )


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
