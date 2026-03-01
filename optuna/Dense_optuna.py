import subprocess
import optuna
import pandas as pd
from pathlib import Path

# paths
TRAIN_PATH = "data/TrainFull.csv"
#MISS_PATH = "data/TestMiss_60.0pct.csv"
MISS_PATH = "data/TestMiss_80.0pct.csv"
TEST_FULL_PATH = "data/TestFull.csv"

RUN_MODEL_NAME = "Dense_at_80"

BASE_DIR = Path("optuna") / "optuna_runs" / RUN_MODEL_NAME
BASE_DIR.mkdir(exist_ok=True)

# objective
def objective(trial):
    hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512, 1024])
    bottleneck_dim = trial.suggest_categorical("bottleneck_dim", [8, 16, 32, 64])
    lr = trial.suggest_categorical("lr", [1e-3, 1e-4])
    activation = trial.suggest_categorical("activation", ["relu", "elu", "linear"])
    l2_factor = trial.suggest_categorical("l2_factor", [0.0, 1e-4, 1e-3, 1e-2])
    mult_1 = trial.suggest_float("mult_1", 1.0, 3.0, step=0.1)
    mult_2 = trial.suggest_float("mult_2", 1.0, 3.0, step=0.1)

    trial_dir = BASE_DIR / f"trial_{trial.number}"
    print(trial_dir)
    trial_dir.mkdir(exist_ok=True)

    save_model = "model.keras"

    cmd = [
        "python", "Dense_imputation.py", "train-and-eval",
        "--train-path", str(TRAIN_PATH),
        "--test-path", str(MISS_PATH),
        "--test-full-path", str(TEST_FULL_PATH), 
        "--save-model", str(save_model),
        "--hidden-dim", str(hidden_dim), 
        "--bottleneck-dim", str(bottleneck_dim), 
        "--val-split", "0.2",
        "--batch-size", "128",
        "--activation", str(activation),
        "--no-gpu",
        "--lr", str(lr), 
        "--results-dir", str(trial_dir),
        "--l2-factor", str(l2_factor),
        "--mult-1", str(mult_1),
        "--mult-2", str(mult_2), 
        "--export-model-name", RUN_MODEL_NAME
    ]

    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"trial {trial.number} failed")

    metrics_path = trial_dir / "train_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError("train_metrics.csv not found")

    df = pd.read_csv(metrics_path)
    f1_macro = float(df["val_f1_macro"].iloc[0])
    val_loss_best = float(df["val_loss_best"].iloc[0])
    best_epoch = int(df["best_epoch"].iloc[0])

    trial.set_user_attr("best_epoch", best_epoch)
    trial.set_user_attr("val_loss_best", val_loss_best)

    return f1_macro


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    print("best trial")
    print("trial number:", study.best_trial.number)
    print("objective value (f1 macro on validation):", study.best_value)
    print("best epoch:", study.best_trial.user_attrs["best_epoch"])
    print("best val loss:", study.best_trial.user_attrs["val_loss_best"])

    print("params:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")