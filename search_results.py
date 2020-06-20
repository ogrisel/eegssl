import optuna
import pandas as pd
import hiplot as hip


pd.set_option("max_rows", 100)


study = optuna.create_study(
    study_name="relative_positioning_physionet",
    direction="maximize",
    storage="sqlite:///optuna_eegssl.db",
    load_if_exists=True,
)
df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
df.sort_values("value", ascending=False).head(25)


hip.Experiment.from_dataframe(df).display()
