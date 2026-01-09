# utils.py
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import typer
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Tuple, Optional

# ------------------------
# Reading files
# ------------------------
def read_csv_array(path: str) -> Optional[np.ndarray]:
    """Read CSV and return float32 array, no header."""
    p = Path(path)
    if not p.exists():
        typer.echo(f"File not found: {path}.")
        return None
    return pd.read_csv(p, header=None).values.astype(np.float32)


def read_maf(maf_path: str) -> Optional[np.ndarray]:
    """Read MAF CSV, drop last column, return float64 array."""
    p = Path(maf_path)
    if not p.exists():
        print(f"[ERROR] File not found: {maf_path}")
        return None

    try:
        dataMAF = pd.read_csv(p, header=None)
        dataMAF = dataMAF.drop(dataMAF.columns[-1], axis=1)
        dataMAF = dataMAF.values.astype(np.float64)
        print(f"[INFO] Loaded MAF '{maf_path}' shape {dataMAF.shape}")
        return dataMAF
    except Exception as e:
        print(f"[ERROR] Failed to read MAF: {e}")
        return None

# ------------------------
# Data balancing
# ------------------------
def prepare_balanced_mask(
    dataTRAIN: np.ndarray,
    dataMISS: Optional[np.ndarray] = None,
    mult_1: float = 2.3,
    mult_2: float = 2.6,
    seed: int = 222
) -> Tuple[np.ndarray, float, float, float]:
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
# Dataset classes
# ------------------------
class MaskedDataset(tf.keras.utils.Sequence):
    """Wrap x and mask into batches for Keras."""
    def __init__(self, x: np.ndarray, mask: np.ndarray, batch_size: int, **kwargs):
        super().__init__(**kwargs)
        self.x = np.array(x, dtype=np.float32)
        self.mask = np.array(mask, dtype=np.float32)
        self.batch_size = batch_size
        self.n_samples = len(self.x)

    def __len__(self) -> int:
        return (self.n_samples + self.batch_size - 1) // self.batch_size

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        start = idx * self.batch_size
        end = min(start + self.batch_size, self.n_samples)
        x_batch = self.x[start:end]
        mask_batch = self.mask[start:end]
        # combine x and mask along last axis
        y_combined = np.stack([x_batch, mask_batch], axis=-1)
        return x_batch, y_combined
    
# ------------------------
# Loss classes
# ------------------------
class MaskedCELoss(tf.keras.losses.Loss):
    """Masked categorical crossentropy with per-class weights."""
    def __init__(self, weight=1.0, weight_0=1.0, weight_1=1.0, weight_2=1.0, name="masked_ce_loss"):
        super().__init__(name=name)
        self.weight = weight
        self.weight_0 = weight_0
        self.weight_1 = weight_1
        self.weight_2 = weight_2

    def call(self, y_true_with_mask, y_pred):
        y_true = y_true_with_mask[..., 0]
        mask_batch = y_true_with_mask[..., 1]
        y_true = tf.cast(y_true, tf.int32)

        mask_weighted = mask_batch * (1. - self.weight) + self.weight
        mask_weighted_genotypes = (
            self.weight_0 * tf.cast(y_true == 0, tf.float32) +
            self.weight_1 * tf.cast(y_true == 1, tf.float32) +
            self.weight_2 * tf.cast(y_true == 2, tf.float32)
        )

        ce = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        weighted_ce = mask_weighted * mask_weighted_genotypes * ce
        return tf.reduce_mean(weighted_ce)

    def get_config(self):
        return {
            "weight": self.weight,
            "weight_0": self.weight_0,
            "weight_1": self.weight_1,
            "weight_2": self.weight_2,
            "name": self.name
        }

class MultiClassFocalLoss(tf.keras.losses.Loss):
    """Multi-class focal loss with per-class weights and masking."""
    def __init__(self, weight=1.0, weight_0=1.0, weight_1=1.0, weight_2=1.0, gamma=2.0, name="multi_class_focal_loss"):
        super().__init__(name=name)
        self.weight = weight
        self.weight_0 = weight_0
        self.weight_1 = weight_1
        self.weight_2 = weight_2
        self.gamma = gamma

    def call(self, y_true_with_mask, y_pred):
        y_true = y_true_with_mask[..., 0]
        mask_batch = y_true_with_mask[..., 1]

        y_true = tf.cast(y_true, tf.int32)
        y_true_oh = tf.one_hot(y_true, depth=3)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)

        alpha = tf.constant([self.weight_0, self.weight_1, self.weight_2], dtype=tf.float32)
        alpha_t = tf.reduce_sum(alpha * y_true_oh, axis=-1)

        p_t = tf.reduce_sum(y_true_oh * y_pred, axis=-1)
        focal_term = tf.pow(1.0 - p_t, self.gamma)

        ce = -tf.reduce_sum(y_true_oh * tf.math.log(y_pred), axis=-1)
        loss = alpha_t * focal_term * ce

        mask_weighted = mask_batch * (1. - self.weight) + self.weight
        loss = mask_weighted * loss

        return tf.reduce_mean(loss)

    def get_config(self):
        return {
            "weight": self.weight,
            "weight_0": self.weight_0,
            "weight_1": self.weight_1,
            "weight_2": self.weight_2,
            "gamma": self.gamma,
            "name": self.name
        }
       
# ------------------------
# Train/val split
# ------------------------
def create_train_val_datasets(
    x: np.ndarray,
    mask: np.ndarray,
    batch_size: int,
    val_split: float = 0.2,
    seed: int = 222,
) -> Tuple[MaskedDataset, Optional[MaskedDataset]]:
    """
    Split data into training and validation (with shuffling) sets and wrap them in MaskedDataset.

    Args:
        x: np.ndarray, shape [n_samples, n_snps]
        mask: np.ndarray, shape [n_samples, n_snps]
        batch_size: int
        val_split: float, fraction of data to use for validation (0.0 = no val)

    Returns:
        train_ds: MaskedDataset
        val_ds: MaskedDataset or None if val_split=0.0
    """
    # number of samples (rows)
    n_samples = x.shape[0]
    n_val = int(n_samples * val_split)

    if val_split > 0.0 and n_val > 0:
        # shuffle rows only
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n_samples)

        # apply the same row permutation to x and mask
        x_shuffled = x[idx]
        mask_shuffled = mask[idx]

        # split after shuffling
        x_train = x_shuffled[:-n_val]
        x_val = x_shuffled[-n_val:]
        mask_train = mask_shuffled[:-n_val]
        mask_val = mask_shuffled[-n_val:]

        val_ds = MaskedDataset(x_val, mask_val, batch_size)
    else:
        x_train = x
        mask_train = mask
        val_ds = None

    train_ds = MaskedDataset(x_train, mask_train, batch_size)
    return train_ds, val_ds

# ------------------------
# Other
# ------------------------
def compute_masked_maf(self, dataMAF, mask_miss, maf_threshold=0.05):
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

def maf_stratified_metrics(self, trueMISS, discrete_predict, dataMAF, mask_missing):
        """
        Compute genotype performance stratified by MAF threshold
        """
        # --- Step 1. Compute SNP rarity flags (0 = common, 1 = rare)
        maf_groups = compute_masked_maf(dataMAF, mask_missing, maf_threshold=self.MAF_threshold)
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


class F1MacroCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_data, mask_data):
        super().__init__()
        self.val_data = val_data        
        self.mask_data = mask_data      

    def on_epoch_end(self, epoch, logs=None):
        # get predictions on validation data
        y_pred = self.model.predict(self.val_data, verbose=0)
        
        # prepare true labels and masked positions
        y_true = self.val_data  # assuming val_data has real labels
        mask = self.mask_data.astype(bool)

        # only consider masked positions
        y_true_masked = y_true[mask]
        y_pred_masked = np.argmax(y_pred, axis=-1)[mask]

        f1_macro = f1_score(y_true_masked, y_pred_masked, average='macro')
        print(f"\nEpoch {epoch+1}: F1 macro = {f1_macro:.4f}\n")

        if logs is not None:
            logs['f1_macro'] = f1_macro