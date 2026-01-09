"""
Dense_imputation.py
"""

import os
import warnings
from pathlib import Path
from datetime import datetime

os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import typer
import numpy as np
import csv
import tensorflow as tf

from keras.layers import Input, Dense, Softmax, Reshape
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import AdamW
from keras.regularizers import l2
from tqdm.keras import TqdmCallback
from sklearn.metrics import classification_report, accuracy_score, f1_score

from utils import (
    read_csv_array, read_maf, prepare_balanced_mask,
    MaskedCELoss, MultiClassFocalLoss,
    create_train_val_datasets, maf_stratified_metrics,
    F1MacroCallback
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ==========================
# TensorFlow & GPU setup
# ==========================
print("TensorFlow version:", tf.__version__)
# print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
# print("Devices:", tf.config.list_physical_devices())

# ==========================
# Typer setup
# ==========================
app = typer.Typer(
    add_completion=True,
    rich_markup_mode="rich",
    help=""" Dense Autoencoder for SNP Imputation

 python Dense_imputation.py --help

 python Dense_imputation.py train --help

 python Dense_imputation.py eval --help

 python Dense_imputation.py train-and-eval --help

"""
)
	
# ------------------------
# Autoencoder class
# ------------------------
class SNP_Autoencoder:
    def __init__(self, num_snps, mask_miss=None,  weight=0.0, weight_0=1.0, weight_1=1.0, weight_2=1.0, lr=1e-4,
                 hidden_dim=256, bottleneck_dim=64, gamma=2.0, activation="relu", l2_factor=1e-4, loss="ce",
                 export_model_name="DenseAE", mult_1=2.3, mult_2=2.6):
        self.num_snps = num_snps
        self.mask_miss = mask_miss

        self.weight = float(weight)
        self.weight_0 = float(weight_0)
        self.weight_1 = float(weight_1)
        self.weight_2 = float(weight_2)
        self.gamma = float(gamma)

        self.mult_1 = mult_1
        self.mult_2 = mult_2

        self.hidden_dim = int(hidden_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.lr = lr
        self.l2_factor = l2_factor

        self.loss = loss
        self.activation = activation
        self.export_model_name = export_model_name

        self.model = self.build_model()
        print(f"[MODEL INFO] {type(self.model).__name__} | params = {self.model.count_params():,}")
        self.compile()
        self.loss_history = None
        self.snp_means = None

    def compile(self):
        opt = AdamW(learning_rate=self.lr)
        if self.loss == "ce":
            self.model.compile(
                optimizer=opt, loss=MaskedCELoss(
                    weight=self.weight, weight_0=self.weight_0,             weight_1=self.weight_1, weight_2=self.weight_2
                )
            )
        else:
            self.model.compile(
                optimizer=opt, loss=MultiClassFocalLoss(
                    weight=self.weight, weight_0=self.weight_0, weight_1=self.weight_1,             weight_2=self.weight_2, gamma=self.gamma
                )
            )

    def build_model(self):
        inp = Input(shape=(self.num_snps,))

        # encoder
        x = Dense(self.hidden_dim, activation=self.activation, kernel_regularizer=l2(self.l2_factor))(inp)
        x = Dense(self.hidden_dim // 2, activation=self.activation, kernel_regularizer=l2(self.l2_factor))(x)

        # bottleneck
        encoded = Dense(self.bottleneck_dim, activation=self.activation)(x)

        # decoder
        x = Dense(self.hidden_dim // 2, activation=self.activation)(encoded)
        x = Dense(self.hidden_dim, activation=self.activation)(x)

        # output logits
        x = Dense(self.num_snps * 3, activation="linear")(x)
        logits = Reshape((self.num_snps, 3))(x)
        probs = Softmax(axis=-1)(logits)

        return Model(inputs=inp, outputs=probs)

    def train(self, x, batch_size, val_split, model_path=None, epochs=1000, results_dir=None, save_means_path=None):
        self.snp_means = np.mean(x, axis=0)

        train_ds, val_ds = create_train_val_datasets(
            x=x, mask=self.mask_miss, batch_size=batch_size, val_split=val_split
        )

        f1_cb = F1MacroCallback(
            val_data=val_ds.x, mask_data=val_ds.mask
        )

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        name = Path(model_path).stem if model_path else "dense_ae"

        if results_dir is None:
            result_dir = Path("results") / f"model_DenseAE_{timestamp}_{name}"
        else:
            result_dir = Path(results_dir)

        result_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_cb = ModelCheckpoint(
            filepath=result_dir / "checkpoint_best.keras", monitor="val_loss", save_best_only=True, mode="min"
        )

        earlystop_cb = EarlyStopping(
            monitor="val_loss", patience=4, min_delta=1e-2, restore_best_weights=True
        )

        history = self.model.fit(
            train_ds, validation_data=val_ds, epochs=epochs, verbose=0, callbacks=[checkpoint_cb, earlystop_cb, TqdmCallback(verbose=1), f1_cb]
        )

        model_save_path = result_dir / f"{name}.keras"
        self.model.save(model_save_path)

        np.save(result_dir / save_means_path, self.snp_means)

        # save training parameters
        params = {
            "timestamp": timestamp,
            "provided_name": name,
            "export_model_name": str(self.export_model_name),
            "loss": self.loss, 
            "num_snps": int(self.num_snps),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "val_split": float(val_split),
            "weight": float(self.weight),
            "weight_0": float(self.weight_0),
            "weight_1": float(self.weight_1),
            "weight_2": float(self.weight_2),
            "mult_1": float(self.mult_1),
            "mult_2": float(self.mult_2),
            "gamma": float(self.gamma),
            "hidden_dim": int(self.hidden_dim),
            "bottleneck_dim": int(self.bottleneck_dim),
            "activation": str(self.activation),
            "lr": float(self.lr),
            "l2_factor": float(self.l2_factor),
            "learning_rate": getattr(self.model.optimizer, 'learning_rate', 'unknown'),
            "model_params": self.model.count_params()
        }
        with open(result_dir / 'params.txt', 'w') as f:
            for k, v in params.items():
                f.write(f"{k}: {v}\n")

        self.loss_history = history.history.get("loss", [])

        # save best val loss & F1 macro 
        # print(history.history.keys())

        best_val_loss = min(history.history["val_loss"])
        best_epoch = int(np.argmin(history.history["val_loss"]))
        f1_at_best_loss = history.history["f1_macro"][best_epoch]
        csv_path = Path(result_dir) / "train_metrics.csv"
        with open(csv_path, "w") as f:
            f.write("model,model_path,val_loss_best,val_f1_macro,best_epoch\n")
            f.write(f"{str(self.export_model_name)},{str(model_save_path)},{best_val_loss},{f1_at_best_loss},{best_epoch}\n")

        return history, str(result_dir)

    def eval(self, data_missing, data_full, data_maf=None, snp_means=None, model_path=None, save_parent_dir=None, dataset_name=""):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_dir = Path(save_parent_dir) if save_parent_dir else Path("results")
        save_dir.mkdir(parents=True, exist_ok=True)

        # load model
        model_to_use = load_model(model_path, compile=False) if model_path else self.model
        n_params = model_to_use.count_params()

        # snp means
        if snp_means is None:
            if self.snp_means is not None:
                snp_means = self.snp_means
            else:
                raise ValueError("No snp_means provided or in object.")

        # fill missing positions
        mask_missing = (data_missing == 3.)
        data_missing_filled = np.where(mask_missing, snp_means, data_missing)

        predicted = model_to_use.predict(data_missing_filled, verbose=0)

        mask_missing_bool = mask_missing.astype(bool)
        predict_missing = predicted[mask_missing_bool]
        true_missing = data_full[mask_missing_bool]

        discrete_predict = np.argmax(predict_missing, axis=-1).astype(int)
        all_labels = true_missing.astype(int)

        print("\nClassification Report for missing SNPs:")
        print(classification_report(all_labels, discrete_predict, digits=3, zero_division=0))

        acc = accuracy_score(all_labels, discrete_predict)
        f1_micro = f1_score(all_labels, discrete_predict, average="micro")
        f1_macro = f1_score(all_labels, discrete_predict, average="macro")
        f1_per_class = f1_score(all_labels, discrete_predict, labels=[0, 1, 2], average=None)

        print(f"Model params: {n_params}")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1 (micro): {f1_micro:.4f}")
        print(f"F1 (macro): {f1_macro:.4f}")
        print(f"F1 per class [0,1,2]: {f1_per_class}")

        if data_maf is not None:
            metrics = maf_stratified_metrics(
                all_labels, discrete_predict, data_maf, mask_missing
            )
            print(metrics["report"])

        csv_path = save_dir / "snp_eval_metrics.csv"
        file_exists = csv_path.exists()

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(
                    ["model", "model_path", "date", "dataset", "f1_micro", "f1_macro", "f1_0", "f1_1", "f1_2", "n_params"]
                )
            writer.writerow([
                str(self.export_model_name), str(model_path), timestamp, dataset_name, f1_micro, f1_macro, f1_per_class[0], f1_per_class[1], f1_per_class[2], n_params
            ])

        return {
            "predict_missing": predict_missing, "true_missing": true_missing, "discrete_predict": discrete_predict
        }


# ------------------------
# CLI commands
# ------------------------
@app.command()
def train(
    train_path: str = typer.Option(..., help="CSV path for training genotypes (no missing values)."),
    miss_path: str = typer.Option(None, help="(Optional) CSV path for masked genotypes (3 == missing), used to create mask."),
    save_model: str = typer.Option("dense_ae_best.keras", help="Path to save best model."),
    save_means: str = typer.Option("snp_means.npy", help="Path to save SNP column means (numpy .npy)."),
    epochs: int = typer.Option(1000, help="Number of training epochs."),
    batch_size: int = typer.Option(16, help="Batch size."),
    weight: float = typer.Option(0.0, help="If 0 then only masked positions are considered in loss."),
    lr: float = typer.Option(1e-4, help="Learning rate."),
    mult_1: float = typer.Option(2.3, help="Maximum oversampling for genotype 1 in mask."),
    mult_2: float = typer.Option(2.6, help="Maximum oversampling for genotype 2 in mask."),
    hidden_dim: int = typer.Option(256, help="Hidden layer dimension."),
    bottleneck_dim: int = typer.Option(64, help="Bottleneck dimension."),
    gpu: bool = typer.Option(True, help="Whether to use GPU (default True). If False, forces CPU training."),
    gamma: float = typer.Option(2.0, help="Gamma for focal loss"),
    activation: str = typer.Option("relu", help="Activation function"),
    val_split: float = typer.Option(0.2, help="Validation split"),
    results_dir: str = typer.Option(None, help="Folder to save result to if different than results/."),
    l2_factor: float = typer.Option(1e-4, help="L2 regularization factor"),
    loss: str = typer.Option("ce", help="Loss function (ce or focal)"),
    export_model_name: str = typer.Option("DenseAE", help="Model name for saving in output csv results.")
):
    # =======================
    # GPU / CPU setup
    # =======================
    if gpu:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for g in gpus:
                tf.config.experimental.set_memory_growth(g, True)
            typer.secho(f"[INFO] Using GPU(s): {len(gpus)}", fg=typer.colors.BRIGHT_GREEN)
        else:
            typer.secho("[INFO] No GPU detected; using CPU.", fg=typer.colors.BRIGHT_YELLOW)
    else:
        tf.config.set_visible_devices([], "GPU")
        typer.secho("[INFO] GPU disabled; training on CPU.", fg=typer.colors.BRIGHT_YELLOW)

    # =======================
    # Read data
    # =======================
    dataTRAIN = read_csv_array(train_path)
    dataMISS = read_csv_array(miss_path) if miss_path else None

    if dataTRAIN is None:
        raise typer.BadParameter("Missing required input files (--train-path).")

    num_snps = dataTRAIN.shape[1]
    typer.secho(f"[INFO] Train data shape: {dataTRAIN.shape}", fg=typer.colors.BRIGHT_GREEN)

    # =======================
    # Mask creation
    # =======================
    typer.secho("[INFO] Mask creation...", fg=typer.colors.BRIGHT_GREEN)
    mask_miss, w0, w1, w2 = prepare_balanced_mask(
        dataTRAIN, dataMISS, mult_1=mult_1, mult_2=mult_2
    )

    # =======================
    # Train model
    # =======================
    typer.secho("[INFO] Starting training...", fg=typer.colors.BRIGHT_GREEN)
    snp_ae = SNP_Autoencoder(
        num_snps=num_snps, mask_miss=mask_miss, weight=weight, weight_0=w0, weight_1=w1, weight_2=w2,
        hidden_dim=hidden_dim, bottleneck_dim=bottleneck_dim, lr=lr, gamma=gamma, activation=activation,
        l2_factor=l2_factor, loss=loss, export_model_name=export_model_name, mult_1=mult_1,
        mult_2=mult_2
    )

    _, run_dir = snp_ae.train(
        dataTRAIN, model_path=save_model, epochs=epochs, batch_size=batch_size, save_means_path=save_means, val_split=val_split, results_dir=results_dir
    )

    typer.secho(f"[SUCCESS] Training finished. Artifacts in {run_dir}", fg=typer.colors.BRIGHT_CYAN, bold=True)
    return str(run_dir)

@app.command()
def eval(
    test_path: str = typer.Option(..., help="CSV path for masked genotypes (3 == missing)"),
    full_path: str = typer.Option(..., help="CSV path for full genotypes (ground truth)"),
    maf_path: str = typer.Option(None, help="[Optional] MAF scores for evaluation."),
    model_path: str = typer.Option("dense_ae_best.keras", help="Path to trained model (.keras)"),
    means_path: str = typer.Option(None, help="Path to SNP means (.npy)."),
    eval_parent_dir: str = typer.Option(None, help="Optional parent directory to place eval outputs."),
    export_model_name: str = typer.Option("DenseAE", help="Model name for saving in output csv results."),
):
    dataMISS = read_csv_array(test_path)
    dataFULL = read_csv_array(full_path)

    if dataMISS is None or dataFULL is None:
        raise typer.BadParameter("Missing required input files.")

    dataMAF = read_maf(maf_path) if maf_path else None
    num_snps = dataMISS.shape[1]

    typer.secho(f"Eval data shapes: {dataMISS.shape}, {dataFULL.shape}", fg=typer.colors.BRIGHT_GREEN)

    if means_path:
        snp_means = np.load(means_path)
        typer.secho(f"[INFO] Loaded SNP means from {means_path}", fg=typer.colors.BRIGHT_GREEN)
    else:
        typer.secho("[INFO] No means provided; using 1.5", fg=typer.colors.BRIGHT_YELLOW)
        snp_means = np.ones(num_snps) * 1.5

    snp_ae = SNP_Autoencoder(num_snps=num_snps, export_model_name=export_model_name)
    snp_ae.eval(
        data_missing=dataMISS, data_full=dataFULL, data_maf=dataMAF, snp_means=snp_means, model_path=model_path, save_parent_dir=eval_parent_dir, dataset_name=test_path,
    )

@app.command("train-and-eval")
def train_and_eval(
    train_path: str = typer.Option(...),
    test_path: str = typer.Option(...),
    test_full_path: str = typer.Option(...),
    maf_path: str = typer.Option(None),
    save_model: str = typer.Option("dense_ae_best.keras"),
    save_means: str = typer.Option("snp_means.npy"),
    epochs: int = typer.Option(1000),
    batch_size: int = typer.Option(16),
    weight: float = typer.Option(0.0),
    lr: float = typer.Option(1e-4),
    mult_1: float = typer.Option(2.3),
    mult_2: float = typer.Option(2.6),
    hidden_dim: int = typer.Option(256),
    bottleneck_dim: int = typer.Option(64),
    gpu: bool = typer.Option(True),
    gamma: float = typer.Option(2.0),
    val_split: float = typer.Option(0.2),
    activation: str = typer.Option("relu", help="Activation function"),
    results_dir: str = typer.Option(None, help="Folder to save result to if different than results/."),
    l2_factor: float = typer.Option(1e-4, help="L2 regularization factor"),
    loss: str = typer.Option("ce", help="Loss function (ce or focal)"),
    export_model_name: str = typer.Option("DenseAE", help="Model name for saving in output csv results."),
):
    run_dir = train(
        train_path=train_path, miss_path=test_path, save_model=save_model, save_means=save_means, epochs=epochs,
        batch_size=batch_size, weight=weight, lr=lr, mult_1=mult_1, mult_2=mult_2, hidden_dim=hidden_dim,
        bottleneck_dim=bottleneck_dim, gpu=gpu, gamma=gamma, val_split=val_split, results_dir=results_dir,
        activation=activation, l2_factor=l2_factor, loss=loss, export_model_name=export_model_name
    )

    eval(
        test_path=test_path, full_path=test_full_path, maf_path=maf_path, model_path=str(Path(run_dir) / save_model),
        means_path=str(Path(run_dir) / "snp_means.npy"), eval_parent_dir=run_dir, export_model_name=export_model_name
    )

if __name__ == "__main__":
    app()