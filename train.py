# %%
from itertools import islice
from time import time

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt  # noqa
import mne
from mne.datasets.sleep_physionet.age import fetch_data


SAMPLING_FREQ = 100  # Hz


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
class PairGenerator:
    """Generate random pairs of windows of EEG data"""

    def __init__(
        self,
        edf_data_chunks,
        window_size=30 * SAMPLING_FREQ,
        tau_pos=100 * SAMPLING_FREQ,
        tau_neg=100 * SAMPLING_FREQ,
        pos_overlap=False,
        pairs_per_chunk=1000,
        preload=False,
        seed=0,
        scale="auto",  # 1e-5,
    ):
        if pos_overlap:
            assert tau_pos >= window_size
        self.data_chunks = edf_data_chunks
        self.window_size = window_size
        self.tau_pos = tau_pos
        self.tau_neg = tau_neg
        self.pos_overlap = pos_overlap
        self.pairs_per_chunk = pairs_per_chunk
        self.seed = seed
        self.preload = preload
        self.scale = scale
        if self.preload:
            self.all_chunks = [
                self._load_eeg_chunk(i) for i in range(len(edf_data_chunks))
            ]
            self.channel_size = self.all_chunks[0].shape[1]
        else:
            self.channel_size = self._load_eeg_chunk(0).shape[1]

    def _load_eeg_chunk(self, chunk_idx):
        eeg_chunk = self.data_chunks[chunk_idx].get_data(picks="eeg")
        eeg_chunk = np.ascontiguousarray(
            eeg_chunk.T, dtype=np.float32
        )  # shape=(timesteps, channels)
        if self.scale == "auto":
            scale = eeg_chunk.std(axis=0).mean()
            eeg_chunk /= scale
        elif self.scale is not None:
            eeg_chunk /= self.scale
        return eeg_chunk

    def _generate_pairs_metadata(self):
        rng = np.random.RandomState(self.seed)
        while True:
            chunk_idx = rng.randint(0, len(self.data_chunks))
            if self.preload:
                eeg_chunk = self.all_chunks[chunk_idx]
            else:
                eeg_chunk = self._load_eeg_chunk(chunk_idx)
            chunk_length = eeg_chunk.shape[0]

            # Amortize the cost of data loading by generating a large
            # enough number of pairs.
            for _ in range(self.pairs_per_chunk):
                if rng.rand() > 0.5:
                    a_start, b_start = self._generate_pos_pair_idx(rng, chunk_length)
                    label = 1
                else:
                    a_start, b_start = self._generate_neg_pair_idx(rng, chunk_length)
                    label = 0
                yield a_start, b_start, label, eeg_chunk

    def generate_pairs(self):
        for a_start, b_start, label, eeg_chunk in self._generate_pairs_metadata():
            yield (
                (
                    eeg_chunk[a_start : a_start + self.window_size],
                    eeg_chunk[b_start : b_start + self.window_size],
                ),
                label,
            )

    def to_tf_dataset(self):
        return tf.data.Dataset.from_generator(
            pair_gen_train.generate_pairs,
            output_types=((tf.float32, tf.float32), tf.int8),
            output_shapes=(
                (
                    (self.window_size, self.channel_size),
                    (self.window_size, self.channel_size),
                ),
                (),
            ),
        )

    def _generate_pos_pair_idx(self, rng, chunk_length):
        if self.pos_overlap:
            dist_low = 0
        else:
            dist_low = self.window_size
        distance = rng.randint(dist_low, self.tau_pos)
        radius = distance // 2
        remainder = distance - radius
        center = rng.randint(radius, chunk_length - self.window_size - remainder)
        a_start = center - radius
        b_start = center + remainder
        return a_start, b_start

    def _generate_neg_pair_idx(self, rng, chunk_length):
        for _ in range(10):
            a_start = rng.randint(0, chunk_length - self.window_size)
            b_start = rng.randint(0, chunk_length - self.window_size)
            if abs(a_start - b_start) < self.tau_neg:
                continue
            if b_start > a_start:
                return a_start, b_start
            else:
                return b_start, a_start
        raise ValueError(
            f"Cannot sample negative pair: tau_neg={self.tau_neg}"
            f" is to large compared to chunk length: {chunk_length}"
        )


# %%
def run_bench(pairs_per_chunk=1000, total_pairs=10000, preload=False):
    print(f"{pairs_per_chunk=}, {total_pairs=}, {preload=}")
    t0 = time()
    pair_gen_train = PairGenerator(
        raw_train, pairs_per_chunk=pairs_per_chunk, preload=preload
    )
    if preload:
        total_bytes = sum(a.nbytes for a in pair_gen_train.all_chunks)
    else:
        total_bytes = 0
    duration = time() - t0
    print(
        f"Init: {total_bytes / 1e6} MB in {duration:.1f} s:"
        f" {total_bytes / duration / 1e6:.1f} MB/s"
    )

    total_bytes = 0
    t0 = time()
    for (a, b), _label in islice(pair_gen_train.generate_pairs(), total_pairs):
        total_bytes += a.nbytes
        total_bytes += b.nbytes
    duration = time() - t0
    print(
        f"Pairs: {total_bytes / 1e6} MB in {duration:.1f} s:"
        f" {total_bytes / duration / 1e6:.1f} MB/s"
    )


# run_bench(pairs_per_chunk=2000, total_pairs=20000)
# run_bench(pairs_per_chunk=1000, total_pairs=20000)
# run_bench(pairs_per_chunk=500, total_pairs=10000)
# run_bench(pairs_per_chunk=100, total_pairs=20000, preload=True)
# run_bench(pairs_per_chunk=10, total_pairs=20000, preload=True)
# run_bench(pairs_per_chunk=1, total_pairs=20000, preload=True)


# %%
print("Loading data...")
pair_gen_train = PairGenerator(raw_train, pairs_per_chunk=256, preload=True)
pair_gen_test = PairGenerator(raw_test, pairs_per_chunk=256, preload=False)


# %%
print("Wrapping as a TensorFlow Dataset")

with tf.device("/GPU:0"):
    ds_pairs_train = pair_gen_train.to_tf_dataset()
    ds_pairs_test = pair_gen_test.to_tf_dataset()
    validation_data = next(iter(ds_pairs_test.batch(1000)))


# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import SeparableConv1D
from tensorflow.keras.layers import MaxPool1D
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.layers import concatenate
from tensorflow_addons.layers import GELU


def make_encoder_model(
    n_blocks=5,
    base_filters=8,
    activation="relu",
    kernel_size=11,
    pool_strides=3,
    dilation_rate=2,
):
    filters = base_filters
    model = Sequential()
    activation_layer = {"relu": ReLU, "gelu": GELU}[activation]
    for block_idx in range(n_blocks):
        if block_idx == 0:
            input_shape = (pair_gen_train.window_size, pair_gen_train.channel_size)
        model.add(
            Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                input_shape=input_shape,
                dilation_rate=dilation_rate,
            )
        )
        model.add(activation_layer())

        if block_idx < n_blocks - 1:
            model.add(MaxPool1D(strides=pool_strides))
            filters *= 2

    model.add(GlobalAveragePooling1D())
    model.add(LayerNormalization())
    return model


encoder = make_encoder_model()
print(encoder.summary())


# %%


class PairModel(tf.keras.models.Model):
    def __init__(
        self,
        n_blocks=5,
        base_filters=8,
        activation="relu",
        kernel_size=11,
        pool_strides=3,
        dilation_rate=2,
        hidden_size=256,
    ):
        super().__init__()
        activation_layer = {"relu": ReLU, "gelu": GELU}[activation]
        self.encoder = make_encoder_model(
            n_blocks=n_blocks,
            base_filters=base_filters,
            kernel_size=kernel_size,
            pool_strides=pool_strides,
            activation=activation,
            dilation_rate=dilation_rate,
        )
        self.mlp_hidden = Dense(units=hidden_size)
        self.mlp_activation = activation_layer()
        self.mlp_output = Dense(units=1)

    def call(self, inputs, **kwargs):
        a, b = inputs
        a = self.encoder(a, **kwargs)
        b = self.encoder(b, **kwargs)
        x = concatenate([a, b, a - b, a * b], axis=-1)
        x = self.mlp_hidden(x)
        x = self.mlp_activation(x)
        return self.mlp_output(x)


with tf.device("/GPU:0"):
    model = PairModel()


steps_per_epoch = 1000
epochs = 300
lr_schedule = tf.keras.experimental.CosineDecay(
    initial_learning_rate=1e-3, decay_steps=steps_per_epoch * epochs
)
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
# optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, nesterov=True, momentum=0.9)
model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])


# %%

(a, b), _ = next(iter(ds_pairs_train.batch(32)))
model([a, b]).shape

loss, accuracy = model.evaluate(*validation_data)
print(f"{loss=:.4f}, {accuracy=:.4f}")

# %%
batch_size = 32
history = model.fit(
    ds_pairs_train.batch(batch_size).prefetch(100),
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=validation_data,
)


# %%
model.evaluate(ds_pairs_test.batch(32).take(1000))


# %%
tf.saved_model.save(model, "third_model.tf")

# %%
