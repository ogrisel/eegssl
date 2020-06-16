from itertools import islice
from time import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import mne
from mne.datasets.sleep_physionet.age import fetch_data
from sklearn.model_selection import train_test_split


# XXX: un-hardcode me!
SAMPLING_FREQ = 100  # Hz


class RawPhysionetLoader:

    mapping = {
        "EOG horizontal": "eog",
        "Resp oro-nasal": "misc",
        "EMG submental": "misc",
        "Temp rectal": "misc",
        "Event marker": "misc",
    }

    def __init__(
        self,
        test_subjects=0.25,
        preprocessed_data_folder="preprocessed_physionet",
        seed=0,
    ):
        self.data_folder = Path(preprocessed_data_folder)
        self.data_folder.mkdir(exist_ok=True)
        self.train_subject_ids, self.test_subject_ids = train_test_split(
            range(20), random_state=seed,
        )

    def load(self, split="train"):
        if split == "train":
            subject_ids = self.train_subject_ids
        elif split == "test":
            subject_ids = self.test_subject_ids
        else:
            raise ValueError(f"Unkwnown split={split}")
        return self.fetch_and_extract(self.data_folder, subject_ids)

    def fetch_and_extract(self, data_folder, subject_ids):
        prepared_filenames = []
        orig_filenames = fetch_data(subject_ids)
        for psg_file, _hypnogram_file in orig_filenames:
            target_filename = Path(psg_file).name.replace(".edf", ".npy")
            target_filename = data_folder / target_filename
            prepared_filenames.append(target_filename)
            if target_filename.exists():
                continue

            print("Loading PSG file with MNE...")
            raw_edf = mne.io.read_raw_edf(psg_file)
            raw_edf.set_channel_types(self.mapping)

            eeg_array = raw_edf.get_data(picks="eeg")
            eeg_array = np.ascontiguousarray(eeg_array.T, dtype=np.float32)
            scale = eeg_array.std(axis=0).mean()
            eeg_array /= scale

            print(f"Saving scale EEG channel data to {target_filename}...")
            np.save(target_filename, eeg_array, allow_pickle=False)
        return prepared_filenames


class PairGenerator:
    """Generate random pairs of windows of EEG data"""

    def __init__(
        self,
        data_chunks,
        window_size=30 * SAMPLING_FREQ,
        tau_pos=100 * SAMPLING_FREQ,
        tau_neg=100 * SAMPLING_FREQ,
        pos_overlap=False,
        pairs_per_chunk=1000,
        preload=False,
        seed=0,
        scale="auto",
    ):
        if pos_overlap:
            assert tau_pos >= window_size
        self.data_chunks = data_chunks
        self.window_size = window_size
        self.tau_pos = tau_pos
        self.tau_neg = tau_neg
        self.pos_overlap = pos_overlap
        self.pairs_per_chunk = pairs_per_chunk
        self.seed = seed
        self.preload = preload
        self.scale = scale
        if self.preload:
            self.all_chunks = [self._load_chunk(i) for i in range(len(data_chunks))]
            self.channel_size = self.all_chunks[0].shape[1]
        else:
            self.channel_size = self._load_chunk(0).shape[1]

    def _load_chunk(self, chunk_idx):
        return np.load(self.data_chunks[chunk_idx])

    def _generate_pairs_metadata(self):
        rng = np.random.RandomState(self.seed)
        while True:
            chunk_idx = rng.randint(0, len(self.data_chunks))
            if self.preload:
                chunk = self.all_chunks[chunk_idx]
            else:
                chunk = self._load_chunk(chunk_idx)
            chunk_length = chunk.shape[0]

            # Amortize the cost of data loading by generating a large
            # enough number of pairs.
            for _ in range(self.pairs_per_chunk):
                if rng.rand() > 0.5:
                    a_start, b_start = self._generate_pos_pair_idx(rng, chunk_length)
                    label = 1
                else:
                    a_start, b_start = self._generate_neg_pair_idx(rng, chunk_length)
                    label = 0
                yield a_start, b_start, label, chunk

    def generate_pairs(self):
        for a_start, b_start, label, chunk in self._generate_pairs_metadata():
            yield (
                (
                    chunk[a_start : a_start + self.window_size],
                    chunk[b_start : b_start + self.window_size],
                ),
                label,
            )

    def to_tf_dataset(self):
        return tf.data.Dataset.from_generator(
            self.generate_pairs,
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


def run_loader_bench(raw_data, pairs_per_chunk=1000, total_pairs=10000, preload=False):
    print(f"{pairs_per_chunk=}, {total_pairs=}, {preload=}")
    t0 = time()
    pair_gen_train = PairGenerator(
        raw_data, pairs_per_chunk=pairs_per_chunk, preload=preload
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
