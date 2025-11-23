"""
CAE_imputation.py

Training / evaluating a SNP convolutional autoencoder for genotype imputation.
- train mode computes and saves SNP column means (per-SNP means) and the best model
- eval mode can take provided means or use means saved during training

"""
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import typer
import tensorflow as tf
from keras.layers       import Input, Dense, Softmax, Reshape, Flatten, Conv1D, SpatialDropout1D, BatchNormalization, Conv1DTranspose
from keras.callbacks    import EarlyStopping, ModelCheckpoint
from keras.models       import Model, load_model
from keras.optimizers   import AdamW
from keras.regularizers import l2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from tqdm.keras import TqdmCallback
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score
print("TensorFlow version:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
print("Devices:", tf.config.list_physical_devices())
# If GPUs are available, enable memory growth to avoid TF allocating all memory.
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        typer.secho(f"[INFO] Enabled memory growth for {len(gpus)} GPU(s)", fg=typer.colors.BRIGHT_GREEN)
    except Exception as e:
        typer.secho(f"[WARNING] Could not set GPU memory growth: {e}", fg=typer.colors.YELLOW)
else:
    typer.secho("[INFO] No GPU detected; using CPU.", fg=typer.colors.BRIGHT_YELLOW)
#tf.debugging.set_log_device_placement(True)

app = typer.Typer(
    add_completion=True,
    rich_markup_mode="rich",
    help=""" Convolutional Autoencoder for SNP Imputation

Examples:
-----------

Train the model:
  python CAE_imputation.py train \\
    --train-path ../data/genotypes.csv \\
    --test-path ../data/genotypesTest.csv \\
    --full-path ../data/genotypesTestFull.csv \\
    --epochs 1000 \\
    --batch-size 128 \\
    --lr 1e-4 \\
    --model-path autoencoder_best.keras \\
    --means-path snp_means.npy

Validate:
  python CAE_imputation.py val \\
    --test-path ../data/genotypesTest.csv \\
    --full-path ../data/genotypesTestFull.csv \\
    --model-path autoencoder_best.keras \\
    --means-path snp_means.npy \\
    --visualize

Minimal:
  python CAE_imputation.py train --train-path data/genotypes.csv --miss-path data/genotypesTest.csv
  python CAE_imputation.py val --test-path data/genotypesTest.csv --full-path data/genotypesTestFull.csv --means-path snp_means.npy

Combined example:
  python CAE_imputation.py train-and-eval --train-path ../data/genotypes.csv --test-path ../data/genotypesTest.csv --test-full-path ../data/genotypesTestFull.csv

"""
)

def read_csv_array(path):
    p = Path(path)
    if not p.exists():
        typer.echo(f"File not found: {path}.")
        return None
    return pd.read_csv(path, header=None).values.astype(np.float32)

def read_maf(maf_path):
    """
    Reads a MAF CSV file (no header), drops the last column,
    and returns it as a NumPy float64 array.
    """
    p = Path(maf_path)
    if not p.exists():
        print(f"[ERROR] File not found: {maf_path}")
        return None

    try:
        dataMAF = pd.read_csv(p, header=None)
        dataMAF = dataMAF.drop(dataMAF.columns[-1], axis=1)
        dataMAF = dataMAF.values.astype(np.float64)
        print(f"[INFO] Loaded MAF file '{maf_path}' with shape {dataMAF.shape}")
        return dataMAF
    except Exception as e:
        print(f"[ERROR] Failed to read MAF file: {e}")
        return None

def prepare_balanced_mask(dataTRAIN, dataMISS=None, mult_1=2.3, mult_2=2.6, seed=222):
    """
    "Mask engineering" - add extra masked positions to balance the genotype
    distribution among masked entries for better representation of rare classes.

    Args:
        dataTRAIN (np.ndarray): Training genotypes (complete, true values).
        dataMISS (np.ndarray): Masked genotypes (3 == missing). If not provided, random 20% of columns are masked as initial mask.
        mult_1 (float): Multiplier to genotype 1 class in masked positions.
        mult_2 (float): Multiplier to genotype 2 class in masked positions.
        seed (int): Random seed.

    Returns:
        mask_final (tf.Tensor): Boolean-like mask (float32) same shape as dataMISS.
        w0, w1, w2 (float): Inverse-frequency weights of genotypes within masked positions.
    """
    rng = np.random.default_rng(seed)
    # no test data provided -> create fake missing positions - 20% of columns
    if dataMISS is None:
        dataMISS = dataTRAIN.copy()
        n_cols_to_modify = int(0.2 * dataMISS.shape[1])
        cols_to_modify = rng.choice(dataMISS.shape[1], size=n_cols_to_modify, replace=False)
        dataMISS[:, cols_to_modify] = 3
        typer.secho(f"[INFO] Mask not provided, created synthetic mask in {n_cols_to_modify} columns.", fg=typer.colors.CYAN)

    # if test data provided but shape is not the same -> error
    if dataTRAIN.shape[1] != dataMISS.shape[1]:
        raise ValueError("dataTRAIN and dataMISS must have the same number of columns (SNPs).")
    

    # check which columns are missing in test data
    col_missing = np.all(dataMISS == 3, axis=0)
    # prepare minimal mask in train data with those missing positions
    mask_temp = np.zeros_like(dataTRAIN, dtype=bool)
    mask_temp[:, col_missing] = True

    # flatten
    mask_flat = mask_temp.flatten()
    orig_flat = dataTRAIN.flatten()

    # stats 
    num_masked = np.sum(mask_flat)
    total = orig_flat.size
    typer.secho(f"Total masked positions (before mask balancing): {num_masked} / {total} ({num_masked / total * 100:.2f}%)", fg=typer.colors.BRIGHT_MAGENTA)

    masked_vals = orig_flat[mask_flat]
    unique, counts = np.unique(masked_vals, return_counts=True)
    typer.secho("Genotype distribution:", fg=typer.colors.BRIGHT_MAGENTA)
    for u, c in zip(unique, counts):
        typer.secho(f"Genotype {int(u)}: {c} ({c / counts.sum() * 100:.2f}%)")

    # balancing
    c0, c1, c2 = [np.sum(masked_vals == x) for x in [0, 1, 2]]
    t1 = min(c0, int(mult_1 * c1))
    t2 = min(c0, int(mult_2 * c2))
    e1, e2 = max(0, t1 - c1), max(0, t2 - c2)

    new_mask = mask_flat.copy()
    for val, extra in [(1, e1), (2, e2)]:
        if extra > 0:
            candidates = np.where((orig_flat == val) & (~new_mask))[0]
            if len(candidates) > 0:
                chosen = rng.choice(candidates, size=min(extra, len(candidates)), replace=False)
                new_mask[chosen] = True

    # after balancing
    typer.secho("***************************", fg=typer.colors.BRIGHT_MAGENTA)
    num_masked_new = np.sum(new_mask)
    typer.secho(f"Total masked positions (after mask balancing): {num_masked_new} / {total} ({num_masked_new / total * 100:.2f}%)", fg=typer.colors.BRIGHT_MAGENTA)


    masked_vals_new = orig_flat[new_mask]
    unique, counts = np.unique(masked_vals_new, return_counts=True)
    total = counts.sum()
    typer.secho("Genotype distribution:", fg=typer.colors.BRIGHT_MAGENTA)
    for u, c in zip(unique, counts):
        typer.secho(f"Genotype {int(u)}: {c} ({c / counts.sum() * 100:.2f}%)")

    inv_freqs = total / counts
    inv_freqs = inv_freqs / np.mean(inv_freqs)  # normalize so average weight ≈ 1
    typer.secho("Inverse count frequency weights:", fg=typer.colors.BRIGHT_MAGENTA)
    for u, w in zip(unique, inv_freqs):
        typer.secho(f"Genotype {int(u)}: weight = {w:.3f}")

    new_mask = new_mask.reshape(dataTRAIN.shape)
    #new_mask = tf.cast(new_mask, dtype=tf.float32)
    new_mask = new_mask.astype(np.float32)

    return new_mask, inv_freqs[0], inv_freqs[1], inv_freqs[2]

# ------------------------
# Dataset class
# ------------------------
class MaskedDataset(tf.keras.utils.Sequence):
    def __init__(self, x, mask, batch_size):
        super().__init__() 
        self.x = np.array(x, dtype=np.float32)
        self.mask = np.array(mask, dtype=np.float32)
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))

    def __getitem__(self, idx):
        batch_idx = np.arange(idx * self.batch_size, (idx + 1) * self.batch_size)
        batch_idx = batch_idx[batch_idx < len(self.x)]
        x_batch = self.x[batch_idx]
        mask_batch = self.mask[batch_idx]
        # pack y_true and mask into one tensor
        y_combined = np.stack([x_batch, mask_batch], axis=-1)  # shape [batch, num_snps, 2]
        return x_batch, y_combined
    
# ------------------------
# Autoencoder class
# ------------------------
class SNP_Autoencoder:
    def __init__(self, num_snps, mask_miss=None, weight=0.0, weight_0=1.0, weight_1=1.0, weight_2=1.0, lr=1e-4, MAF_threshold=0.05, window_size=15, embed_dim=16):
        self.num_snps = num_snps
        self.mask_miss = mask_miss

        self.weight = float(weight)
        self.weight_0 = float(weight_0)
        self.weight_1 = float(weight_1)
        self.weight_2 = float(weight_2)

        self.window_size = window_size
        self.embed_dim = embed_dim

        self.model = self.build_model()
        self.loss_history = None
        self.snp_means = None
        self.MAF_threshold = MAF_threshold


    def masked_ce(self, y_true_with_mask, y_pred):
        """
        y_true: integer labels in shape (batch, num_snps)
        y_pred: probabilities shape (batch, num_snps, 3)
        Compute sparse categorical crossentropy per SNP and apply weights:
          - mask_weighted: up/down weight missing positions per sample
          - genotype weights: weight_0/1/2
        """
        y_true = y_true_with_mask[..., 0]
        mask_batch = y_true_with_mask[..., 1]
        y_true = tf.cast(y_true, tf.int32)  # [batch, num_snps]
        mask_weighted = mask_batch * (1. - self.weight) + self.weight

        # weights for genotypes 0, 1, 2
        mask_weighted_genotypes = (
            self.weight_0 * tf.cast(y_true == 0, tf.float32) +
            self.weight_1 * tf.cast(y_true == 1, tf.float32) +
            self.weight_2 * tf.cast(y_true == 2, tf.float32)
        )

        # categorical crossentropy per SNP
        ce = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)  # [batch, num_snps]

        # apply both weights
        weighted_ce = mask_weighted * mask_weighted_genotypes * ce

        # average over all SNPs in batch
        loss = tf.reduce_mean(weighted_ce)
        return loss
    
    def multi_class_focal_loss(self, y_true_with_mask, y_pred, gamma=2.0):
        """
        Multi-class focal loss with per-class weights and masking.
        """
        y_true = y_true_with_mask[..., 0]
        mask_batch = y_true_with_mask[..., 1]

        # Convert y_true to int and one-hot encode
        y_true = tf.cast(y_true, tf.int32)
        y_true_oh = tf.one_hot(y_true, depth=3)

        # Clip predictions to avoid log(0)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)

        # Compute per-class alpha weights
        alpha = tf.constant([self.weight_0, self.weight_1, self.weight_2], dtype=tf.float32)
        alpha_t = tf.reduce_sum(alpha * y_true_oh, axis=-1)

        # Compute p_t (prob of true class)
        p_t = tf.reduce_sum(y_true_oh * y_pred, axis=-1)

        # Focal term
        focal_term = tf.pow(1.0 - p_t, gamma)

        # CE term
        ce = -tf.reduce_sum(y_true_oh * tf.math.log(y_pred), axis=-1)

        # Combine: α * (1 - p_t)^γ * CE
        loss = alpha_t * focal_term * ce

        # Apply your SNP mask weighting
        mask_weighted = mask_batch * (1. - self.weight) + self.weight
        loss = mask_weighted * loss

        return tf.reduce_mean(loss)

    def compile(self, lr, loss="ce"):
        opt = AdamW(learning_rate=lr)
        if loss == "ce":
            self.model.compile(optimizer=opt, loss=self.masked_ce)
        else:
            # multi_class_focal_loss
            self.model.compile(optimizer=opt, loss=self.multi_class_focal_loss)

    def train(self, x, model_path=None, epochs=1000, batch_size=64, val_split=0.2, save_means_path=None):
        """
        Train the autoencoder and save results in a timestamped folder under `results/`.

        model_path: optional string used as part of the folder/name (can be a filename).
        save_means_path: optional path to also save `snp_means.npy`
        Returns: (history, result_dir)
        """
        self.snp_means = np.mean(x, axis=0)
        # split train/val
        n_val = int(len(x) * val_split)
        x_train, x_val = x[:-n_val], x[-n_val:]
        mask_train, mask_val = self.mask_miss[:-n_val], self.mask_miss[-n_val:]

        train_ds = MaskedDataset(x_train, mask_train, batch_size)
        val_ds = MaskedDataset(x_val, mask_val, batch_size)

        # prepare results folder per-run to avoid clutter
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        provided_name = "model"
        if model_path is not None:
            try:
                provided_name = Path(model_path).stem
            except Exception:
                provided_name = str(model_path)

        result_dir = Path("results") / f"model_CAE_{timestamp}_{provided_name}"
        result_dir.mkdir(parents=True, exist_ok=True)

        # callbacks: save best model inside the run folder
        checkpoint_cb = ModelCheckpoint(
            filepath=str(result_dir / "checkpoint_best.keras"), monitor='val_loss', save_best_only=True, mode='min'
        )
        earlystop = EarlyStopping(monitor='val_loss', patience=4, min_delta=1e-2, restore_best_weights=True)

        history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            verbose=0,
            callbacks=[checkpoint_cb, earlystop, TqdmCallback(verbose=1)]
        )

        # save final model and artifacts inside the run folder
        model_save_path = result_dir / f"{provided_name}.keras"
        self.model.save(model_save_path)

        # save SNP means
        means_save_path = result_dir / "snp_means.npy"
        if self.snp_means is not None:
            np.save(means_save_path, self.snp_means)
            if save_means_path is not None and Path(save_means_path).resolve() != means_save_path.resolve():
                try:
                    np.save(save_means_path, self.snp_means)
                except Exception:
                    pass

        # write run parameters to params.txt for reproducibility
        params = {
            'timestamp': timestamp,
            'provided_name': provided_name,
            'num_snps': int(self.num_snps),
            'epochs': int(epochs),
            'batch_size': int(batch_size),
            'val_split': float(val_split),
            'weight': float(self.weight),
            'weight_0': float(self.weight_0),
            'weight_1': float(self.weight_1),
            'weight_2': float(self.weight_2),
            'window_size': int(self.window_size),
            'embed_dim': int(self.embed_dim)
        }
        # try to read optimizer lr if available
        try:
            lr = self.model.optimizer.learning_rate
            # convert tensor/ schedule to string/value
            try:
                params['learning_rate'] = float(lr)
            except Exception:
                params['learning_rate'] = str(lr)
        except Exception:
            params['learning_rate'] = 'unknown'

        params_file = result_dir / 'params.txt'
        with open(params_file, 'w') as fh:
            for k, v in params.items():
                fh.write(f"{k}: {v}\n")

        self.loss_history = history.history.get('loss', [])
        return history, str(result_dir)

    def _visual_eval(self, predict_missing, true_missing):
        all_preds = np.argmax(predict_missing, axis=-1).astype(int)
        all_labels = true_missing.astype(int)

        if self.loss_history is not None and len(self.loss_history) > 0:
            plt.figure(figsize=(6,4))
            plt.plot(self.loss_history)
            plt.title('Training loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.show()

        # confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        fig, ax = plt.subplots(figsize=(8, 6))
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        plt.title("Confusion Matrix on missing positions")
        plt.show()

    def build_model(self, activation='relu', dropout=0.2):
        """ Conv1D Autoencoder for SNP imputation. """
        input_layer = Input(shape=(self.num_snps,))
        x = Reshape((self.num_snps, 1))(input_layer)  # Conv expects 3D: (batch, length, channels)

        # --- Encoder ---
        x = Conv1D(self.embed_dim * 4, self.window_size, activation=activation, padding='same')(x)
        x = Conv1D(self.embed_dim * 2, self.window_size, activation=activation, padding='same')(x)
        x = Conv1D(self.embed_dim, self.window_size, activation=activation, padding='same')(x)
        x = BatchNormalization()(x)
        x = SpatialDropout1D(dropout)(x)

        # bottleneck
        x = Flatten()(x)
        encoded = Dense(self.embed_dim * 4, activation=activation, kernel_regularizer=l2(1e-4))(x)

        # --- Decoder ---
        x = Dense(self.num_snps * self.embed_dim, activation=activation)(encoded)
        x = Reshape((self.num_snps, self.embed_dim))(x)
        x = Conv1DTranspose(self.embed_dim * 2, self.window_size, padding='same', activation=activation)(x)
        x = Conv1DTranspose(self.embed_dim * 4, self.window_size, padding='same', activation=activation)(x)

        # Final layer: project back to 3 genotype probabilities per SNP
        logits = Conv1D(3, 1, padding='same', kernel_regularizer=l2(1e-4))(x)
        probs = Softmax(axis=-1)(logits)

        return Model(inputs=input_layer, outputs=probs)
    
    def _compute_masked_maf(self, dataMAF, mask_miss, maf_threshold=0.05):
        """
        rare variant flags of missing SNPs.
        dataMAF[i] = [freq_major, freq_minor] for SNP i.
        mask_miss = binary mask (1 if SNP i is masked/missing, 0 if not).
        maf_threshold = cutoff for rare
        """
        mask_miss = tf.convert_to_tensor(mask_miss, dtype=tf.float32)
        mask_flat = tf.reshape(mask_miss[0], [-1])  # shape (1, n_snps)


        p = np.sum(dataMAF * mask_flat.numpy()[:, None], axis=1)
        p_min = np.min(dataMAF, axis=1)
        valid_mask = p > 0
        p_min_valid = p_min[valid_mask]
        is_rare = (p_min_valid <= maf_threshold).astype(int)

        mafMISS = np.column_stack((p_min_valid, is_rare))
        return mafMISS
    
    def _maf_stratified_metrics(self, trueMISS, discrete_predict, dataMAF, mask_missing):
        """
        Compute genotype performance stratified by MAF threshold)
        """
        # --- Step 1. Compute SNP rarity flags (0 = common, 1 = rare)
        maf_groups = self._compute_masked_maf(dataMAF, mask_missing, maf_threshold=self.MAF_threshold)
        rare_flags = maf_groups[:, 1]
        
        # --- Step 2. Expand rarity flags to match flattened genotype data
        n_individuals = len(trueMISS) // len(rare_flags)
        if len(trueMISS) % len(rare_flags) != 0:
            raise ValueError(f"trueMISS length ({len(trueMISS)}) not divisible by SNP count ({len(rare_flags)}).")
        rare_flags_full = np.tile(rare_flags, n_individuals)

        # --- Step 3. Evaluate separately for rare/common SNPs
        results = {}
        for label, desc in zip([0, 1], ["Common", "Rare"]):
            mask = rare_flags_full == label
            t, p = trueMISS[mask], discrete_predict[mask]

            f1_micro = f1_score(t, p, average='micro')
            f1_macro = f1_score(t, p, average='macro')
            report = classification_report(t, p, digits=3, output_dict=False, zero_division=0)
            cm = confusion_matrix(t, p)

            results[label] = {
                "desc": desc,
                "report": report,
                "f1_micro": f1_micro,
                "f1_macro": f1_macro,
                "confusion_matrix": cm
            }

        return results


    def eval(self, data_missing, data_full, data_maf=None, snp_means=None, visualize=False, model_path=None, save_parent_dir=None):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # If a parent directory is provided, create the eval folder inside it.
        if save_parent_dir is not None:
            save_dir = Path(save_parent_dir) / f"eval_{timestamp}"
        else:
            save_dir = Path("results") / f"eval_{timestamp}"
        save_dir.mkdir(parents=True, exist_ok=True)

        if model_path is not None:
            model_to_use = load_model(model_path, compile=False)
        else:
            model_to_use = self.model

        if snp_means is None:
            if self.snp_means is not None:
                snp_means = self.snp_means
            else:
                raise ValueError("No snp_means provided and none available in object. Provide --means-path or run training first.")

        # fill missing values (3) with snp_means
        mask_missing = (data_missing == 3.)
        data_missing_filled = np.where(mask_missing, snp_means, data_missing)

        # predict
        predicted = model_to_use.predict(data_missing_filled, verbose=0)  # shape (N, num_snps, 3)
        mask_missing_bool = mask_missing.astype(bool)

        # gather predictions and true labels at missing positions
        predict_missing = predicted[mask_missing_bool]  # (num_missing_positions, 3)
        true_missing = data_full[mask_missing_bool]     # (num_missing_positions,)

        discrete_predict = np.argmax(predict_missing, axis=-1).astype(int)

        if visualize:
            self._visual_eval(predict_missing, true_missing)

        # classification metrics
        all_labels = true_missing.astype(int)
        typer.secho("\nClassification Report for missing SNPs:", fg=typer.colors.BRIGHT_MAGENTA)
        print(classification_report(all_labels, discrete_predict, digits=3, zero_division=0))
        correct = (discrete_predict == all_labels).sum()
        total = all_labels.size
        print(f"Correct predictions: {correct} / {total} ({correct / total * 100:.2f}%)")

        f1_micro = f1_score(all_labels, discrete_predict, average='micro')
        f1_macro = f1_score(all_labels, discrete_predict, average='macro')
        f1_per_class = f1_score(all_labels, discrete_predict, average=None)
        acc = accuracy_score(true_missing, discrete_predict)
        report = classification_report(true_missing, discrete_predict, digits=3, zero_division=0)

        typer.echo(typer.style(f"F1 (micro): {f1_micro:.4f}", fg=typer.colors.BRIGHT_CYAN, bold=True))
        typer.echo(typer.style(f"F1 (macro): {f1_macro:.4f}", fg=typer.colors.BRIGHT_CYAN, bold=True))
        typer.echo(typer.style(f"F1 per class [0,1,2]: {f1_per_class}", fg=typer.colors.BRIGHT_CYAN, bold=True))

        # confusion matrix
        cm = confusion_matrix(all_labels, discrete_predict)
        TP = np.diag(cm)
        FP = cm.sum(axis=0) - TP
        FN = cm.sum(axis=1) - TP
        TN = cm.sum() - (FP + FN + TP)
        typer.secho("\n=== Confusion Matrix Details ===", fg=typer.colors.BRIGHT_MAGENTA)
        for i in range(len(TP)):
            print(f"Class {i}: TP={TP[i]}, FP={FP[i]}, FN={FN[i]}, TN={TN[i]}")
        print("\nOverall:")
        print(f"TP={TP.sum()}, FP={FP.sum()}, FN={FN.sum()}, TN={TN.sum()}")

        # --- optional MAF-based analysis ---
        if data_maf is not None:
            typer.secho("\n=== Stratified by MAF ===", fg=typer.colors.BRIGHT_MAGENTA)
            metrics = self._maf_stratified_metrics(true_missing, discrete_predict, data_maf, mask_missing)[1]
            print("Rare SNPs positions:")
            print("  Classification report:")
            print(metrics['report'])
            print(f"  F1-micro : {metrics['f1_micro']:.4f}")
            print(f"  F1-macro : {metrics['f1_macro']:.4f}")
            print("  Confusion matrix:")
            print(metrics["confusion_matrix"])
            print("---------------------------------------------------\n")

        # save main metrics
        csv_path = save_dir / f"{timestamp}_CAEeval_results.csv"


        df = pd.DataFrame({
            "metric": ["accuracy", "f1_micro", "f1_macro", "f1_class_0", "f1_class_1", "f1_class_2", "classification_report"],
            "value": [
                acc,
                f1_micro,
                f1_macro,
                f1_per_class[0],
                f1_per_class[1],
                f1_per_class[2],
                report
            ]
        })

        df.to_csv(csv_path, index=False)


        return {
            "predict_missing": predict_missing,
            "true_missing": true_missing,
            "discrete_predict": discrete_predict
        }

@app.command()
def train(
    train_path: str = typer.Option(..., help="CSV path for training genotypes (no missing values)."),
    miss_path: str = typer.Option(None, help="(Optional) CSV path for masked genotypes (3 == missing), used to create mask."),
    save_model: str = typer.Option("autoencoder_best.keras", help="Path to save best model."),
    save_means: str = typer.Option("snp_means.npy", help="Path to save SNP column means (numpy .npy)."),
    epochs: int = typer.Option(1000, help="Number of training epochs."),
    batch_size: int = typer.Option(128, help="Batch size."),
    weight: float = typer.Option(0.0, help="If 0 then only masked positions are considered in loss."),
    lr: float = typer.Option(1e-4, help="Learning rate."),
    mult_1: float = typer.Option(2.3, help="Maximum oversampling for genotype 1 in mask."),
    mult_2: float = typer.Option(2.6, help="Maximum oversampling for genotype 2 in mask."),
    window_size: int = typer.Option(15, help="Window size for convolutional layers."),
    embed_dim: int = typer.Option(16, help="Dimention of embedding.")
):
    # Read data
    dataTRAIN = read_csv_array(train_path)
    if miss_path is not None:
        dataMISS = read_csv_array(miss_path)
    else:
        dataMISS = None

    if dataTRAIN is None:
        raise typer.BadParameter("Missing required input files (--train-path).")


    num_snps = dataTRAIN.shape[1]
    typer.secho(f"[INFO] Train data shape: {dataTRAIN.shape}", fg=typer.colors.BRIGHT_GREEN)

    # balance masked genotypes
    typer.secho("[INFO] Mask creation...", fg=typer.colors.BRIGHT_GREEN)
    mask_miss, w0, w1, w2 = prepare_balanced_mask(dataTRAIN, dataMISS, mult_1=mult_1, mult_2=mult_2)

    typer.secho("[INFO] Starting training...", fg=typer.colors.BRIGHT_GREEN)
    # prepare model
    snp_ae = SNP_Autoencoder(
        num_snps=num_snps,
        mask_miss=mask_miss,
        weight=weight,
        weight_0=w0, weight_1=w1, weight_2=w2,
        window_size = window_size,
        embed_dim = embed_dim
    )
    snp_ae.compile(lr=lr)
    history, run_dir = snp_ae.train(dataTRAIN, model_path=save_model, epochs=epochs, batch_size=batch_size, save_means_path=save_means)

    typer.secho(f"[INFO] Run artifacts saved to: {run_dir}", fg=typer.colors.BRIGHT_GREEN)
    typer.secho(f"[SUCCESS] Training finished. Saved best model and artifacts in: {run_dir}", fg=typer.colors.BRIGHT_CYAN, bold=True)
    return str(run_dir)

@app.command() 
def eval(
    test_path: str = typer.Option(..., help="CSV path for masked genotypes (3 == missing)"),
    full_path: str = typer.Option(..., help="CSV path for full genotypes (ground truth)"),
    maf_path: str = typer.Option(None, help="[Optional] MAF scores for evaluation."),
    model_path: str = typer.Option("autoencoder_best.keras", help="Path to trained model (.keras)"),
    means_path: str = typer.Option(None, help="Path to SNP means (.npy). If not provided, must run training within same process."),
    visualize: bool = typer.Option(False, help="Show confusion matrix and loss plot"),
    eval_parent_dir: str = typer.Option(None, help="Optional parent directory to place eval outputs (e.g., training run dir). If provided, eval results will be written to <parent>/eval_<timestamp>.")
):
    dataMISS = read_csv_array(test_path)
    dataFULL = read_csv_array(full_path)
    if dataMISS is None or dataFULL is None:
        raise typer.BadParameter("Missing required input files (--test_path, --full_path).")
    if maf_path is not None:
        dataMAF = read_maf(maf_path)
    else:
        dataMAF = None

    num_snps = dataMISS.shape[1]
    typer.secho(f"Eval data shapes: {dataMISS.shape}, {dataFULL.shape}", fg=typer.colors.BRIGHT_GREEN)

    snp_means = None
    if means_path is not None:
        if not Path(means_path).exists():
            raise typer.BadParameter(f"[ERROR] means_path {means_path} does not exist.")
        snp_means = np.load(means_path)
        typer.secho(f"[INFO] Loaded SNP means from {means_path}", fg=typer.colors.BRIGHT_GREEN)
    else:
        typer.secho("[INFO] No means_path provided: using default 1.5 for all SNPs.", fg=typer.colors.BRIGHT_YELLOW)
        snp_means = np.ones(num_snps) * 1.5
        
    typer.secho("[INFO] Predicting on test dataset...", fg=typer.colors.BRIGHT_GREEN)
    snp_ae = SNP_Autoencoder(num_snps=num_snps)
    snp_ae.eval(data_missing=dataMISS, data_full=dataFULL, data_maf=dataMAF, snp_means=snp_means, visualize=visualize, model_path=model_path, save_parent_dir=eval_parent_dir)
    typer.secho("[SUCCESS] Evaluation complete.", fg=typer.colors.BRIGHT_CYAN, bold=True)

@app.command("train-and-eval")
def train_and_eval(
    train_path: str = typer.Option(..., help="CSV path for training genotypes (no missing values)."),
    test_path: str = typer.Option(..., help="CSV path for masked test genotypes (3 == missing)."),
    test_full_path: str = typer.Option(..., help="CSV path for full test genotypes (ground truth)."),
    maf_path: str = typer.Option(None, help="(Optional) MAF file for stratified evaluation."),
    save_model: str = typer.Option("autoencoder_best.keras", help="Path to save best model."),
    save_means: str = typer.Option("snp_means.npy", help="Path to save SNP means (.npy)."),
    epochs: int = typer.Option(1000, help="Number of training epochs."),
    batch_size: int = typer.Option(128, help="Batch size."),
    weight: float = typer.Option(0.0, help="If 0 then only masked positions are considered in loss."),
    lr: float = typer.Option(1e-4, help="Learning rate."),
    mult_1: float = typer.Option(2.3, help="Oversampling multiplier for genotype 1."),
    mult_2: float = typer.Option(2.6, help="Oversampling multiplier for genotype 2."),
    visualize: bool = typer.Option(False, help="Visualize confusion matrix and loss plots."),
    window_size: int = typer.Option(15, help="Window size for convolutional layers."),
    embed_dim: int = typer.Option(16, help="Dimention of embedding.")
):
    """
    Train the SNP autoencoder and immediately evaluate it on a test dataset.
    Combines `train` and `eval` into one command.
    """
    run_dir = train(
        train_path=train_path,
        miss_path=test_path,
        save_model=save_model,
        save_means=save_means,
        epochs=epochs,
        batch_size=batch_size,
        weight=weight,
        lr=lr,
        mult_1=mult_1,
        mult_2=mult_2,
        window_size=window_size,
        embed_dim=embed_dim
    )

    model_path_in_run = Path(run_dir) / Path(save_model).name
    means_path_in_run = Path(run_dir) / "snp_means.npy"

    eval(
        test_path=test_path,
        full_path=test_full_path,
        maf_path=maf_path,
        model_path=str(model_path_in_run),
        means_path=str(means_path_in_run),
        visualize=visualize,
        eval_parent_dir=str(run_dir),
    )

if __name__ == "__main__":
    app()