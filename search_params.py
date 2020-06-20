import calendar
import pickle
import json
from pathlib import Path

import optuna
from optuna.integration import TFKerasPruningCallback
import tensorflow as tf
import matplotlib.pyplot as plt  # noqa

from eegssl.data_loader import PairGenerator
from eegssl.data_loader import RawPhysionetLoader
from eegssl.model import PairModel


BATCH_SIZE = 32
EPOCHS = 5
STEPS_PER_EPOCH = 10000


print("Loading data...")
physionet_loader = RawPhysionetLoader()

pair_gen_train = PairGenerator(
    physionet_loader.load(split="train"), pairs_per_chunk=100, preload=True
)
pair_gen_test = PairGenerator(
    physionet_loader.load(split="test"), pairs_per_chunk=100, preload=True
)


print("Wrapping as a TensorFlow Dataset")
with tf.device("/GPU:0"):
    ds_pairs_train = pair_gen_train.to_tf_dataset()
    ds_pairs_test = pair_gen_test.to_tf_dataset()
    validation_data = ds_pairs_test.batch(256).take(50)


def fit_evaluate(trial):
    with tf.device("/GPU:0"):
        try:
            model = PairModel(
                n_blocks=trial.suggest_int("n_blocks", 1, 6),
                base_filters=trial.suggest_int("base_filters", 8, 32, log=True),
                kernel_size=trial.suggest_int("kernel_size", 3, 11),
                pool_strides=trial.suggest_int("pool_strides", 2, 5),
                dilation_rate=trial.suggest_int("dilation_rate", 1, 5),
                hidden_size=trial.suggest_int("hidden_size", 32, 512, log=True),
                input_shape=(pair_gen_train.window_size, pair_gen_train.channel_size),
            )
        except Exception:
            # Invalid model architecture
            return 0.0

    initial_learning_rate = trial.suggest_loguniform(
        "initial_learning_rate", 1e-5, 1e-2
    )
    lr_schedule = tf.keras.experimental.CosineDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=STEPS_PER_EPOCH * EPOCHS,
    )
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

    # Checkpointing
    utc_timestamp = calendar.timegm(trial.datetime_start.utctimetuple())
    run_name = f"trial-{trial.number:05d}-{utc_timestamp}"
    run_path = Path("checkpoints") / run_name
    checkpoint_path = run_path / "epoch-{epoch:02d}"
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(str(checkpoint_path))

    history = model.fit(
        ds_pairs_train.batch(BATCH_SIZE),
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=EPOCHS,
        validation_data=validation_data,
        callbacks=[checkpoint_cb, TFKerasPruningCallback(trial, "val_accuracy")],
    )
    with open(run_path / "history.pkl", "wb") as f:
        pickle.dump(history.history, f)

    with open(run_path / "params.json", "w") as f:
        json.dump(trial.params, f, sort_keys=True, indent=2)

    _, accuracy = model.evaluate(validation_data)
    return accuracy


study = optuna.create_study(
    study_name="relative_positioning_physionet",
    direction="maximize",
    # pruner=optuna.pruners.PercentilePruner(percentile=5.0),
    storage="sqlite:///optuna_eegssl.db",
    load_if_exists=True,
)
study.optimize(fit_evaluate, n_trials=100)
pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
complete_trials = [
    t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
]
print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
