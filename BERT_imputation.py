"""
BERT_imputation.py
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = "true"

from pathlib import Path
from datetime import date, datetime
import time

import typer
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import (
    BertTokenizer,
    BertTokenizerFast,
    BertForMaskedLM,
    BertConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback
)
from typing import Optional, Tuple, Union
from transformers.modeling_outputs import MaskedLMOutput
from sklearn.metrics import f1_score, accuracy_score, classification_report
from utils import read_csv_array, read_maf, maf_stratified_metrics
import csv
import shutil

# Optional perf import
# try:
#     import psutil
# except Exception:
#     psutil = Nones

app = typer.Typer(
    add_completion=True,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
    help="""
BERT SNP imputation CLI

 python Bert_imputation.py --help

 python Bert_imputation.py train --help

 python Bert_imputation.py eval --help

 python Bert_imputation.py train-and-eval --help
"""
)
today = date.today().strftime("%Y%m%d")

class DictDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}

    def __len__(self):
        return len(self.encodings["input_ids"])

class EpochLoggerCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            # compute current epoch
            epoch = state.global_step / max(1, state.max_steps) * args.num_train_epochs
            logs["epoch"] = epoch

class SNPDataProcessor:
    def __init__(self, genotypes_train=None, genotypes_test_full=None, genotypes_test_missing=None, sentence_size: int = 512, vocab=None,
                 val_fraction=0.2):
        self.sentence_size = sentence_size
        self.vocab = vocab or ["0", "1", "2", "[PAD]", "[MASK]", "[CLS]", "[SEP]"]
        self.tokenizer = None
        self.genotypes_train = genotypes_train
        self.genotypes_test_missing = genotypes_test_missing
        self.genotypes_test_full = genotypes_test_full
        self.missing_cols_ids = None
        self.val_fraction=val_fraction

    @staticmethod
    def row_to_str(row):
        return " ".join(map(str, row))

    def genotype_to_sentences(self, arr):
        if arr is None:
            return None
        n_rows, n_cols = arr.shape
        sentences = []

        for start in range(0, n_cols, self.sentence_size):
            end = min(start + self.sentence_size, n_cols)
            sub = arr[:, start:end]  # shape (n_rows, chunk_len)
            sentences.extend(" ".join(map(str, row.astype(int))) for row in sub)
        return sentences

    def create_tokenizer(self, vocab_file="vocab.txt", fast=True):
        vfile = Path(vocab_file)
        if not vfile.exists():
            with open(vfile, "w") as f:
                for token in self.vocab:
                    f.write(token + "\n")
        if fast:
            self.tokenizer = BertTokenizerFast(vocab_file=str(vfile), do_lower_case=False)
        else:
            self.tokenizer = BertTokenizer(vocab_file=str(vfile), do_lower_case=False)
        typer.secho(f"[INFO] Tokenizer created (fast={fast}) vocab_file={vfile}", fg=typer.colors.BRIGHT_GREEN)
        return self.tokenizer

    def tokenize(self, sentences):
        if sentences is None: return None
        return self.tokenizer(sentences, padding=True, truncation=False, return_tensors="pt")

    def split_dataset(self, dataset, seed=222):
        if dataset is None: return None
        total_len = len(dataset)
        val_len = int(self.val_fraction * total_len)
        train_len = total_len - val_len
        if seed is not None:
            generator = torch.Generator().manual_seed(seed)
            train_dataset, val_dataset = random_split(dataset, [train_len, val_len], generator=generator)
        else:
            train_dataset, val_dataset = random_split(dataset, [train_len, val_len])
        return train_dataset, val_dataset
    
    def analyze_missing(self):
        print("Token counts in training data:")
        print(pd.Series(self.genotypes_train.values.flatten()).value_counts())

        all_three_cols = self.genotypes_test_missing.columns[(self.genotypes_test_missing == 3).all()]
        print("\nColumns with all missing tokens (original test):", list(all_three_cols))
        print("Number of columns with all missing tokens (original test):", len(list(all_three_cols)))
    
    def prepare_datasets(self, vocab_file="vocab.txt"):
        # self.analyze_missing()

        train_sentences = self.genotype_to_sentences(self.genotypes_train)
        test_sentences = self.genotype_to_sentences(self.genotypes_test_missing)
        test_sentences_full = self.genotype_to_sentences(self.genotypes_test_full)

        # mask only if test exists
        masked_sentences = None
        if test_sentences is not None:
            masked_sentences = [
                " ".join("[MASK]" if x == "3" else x for x in sent.split())
                for sent in test_sentences
            ]

        self.create_tokenizer(vocab_file = vocab_file)
        tokenized_train = self.tokenize(train_sentences)
        tokenized_test_masked = self.tokenize(masked_sentences)
        tokenized_test_full = self.tokenize(test_sentences_full)

        self.missing_cols_ids = None
        if tokenized_test_masked is not None:
            self.missing_cols_ids = (
                (tokenized_test_masked["input_ids"] == self.tokenizer.mask_token_id)
                .all(dim=0)
                .nonzero(as_tuple=True)[0]
                .tolist()
            )
            typer.secho(
                f"[INFO] Missing columns count: {len(self.missing_cols_ids)}",
                fg=typer.colors.BRIGHT_MAGENTA
            )

        return tokenized_train, tokenized_test_masked, tokenized_test_full


class SNPColumnBalancingMaskingCollator(DataCollatorForLanguageModeling):
    """
    Vectorized collator that:
    - always masks `masked_positions`
    - computes counts of 0/1/2 at masked_positions
    - computes extra_1 / extra_2 to balance classes 1 and 2 vs class 0
    - for each sample, randomly masks up to extra_1 tokens==1 and extra_2 tokens==2
    """
    def __init__(self, tokenizer, masked_positions, mult_1=1.8, mult_2=2.0):
        super().__init__(tokenizer=tokenizer, mlm=True, mlm_probability=0.0)
        self.mask_token_id = tokenizer.mask_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.mult_1 = mult_1
        self.mult_2 = mult_2
        # keep masked_positions as a list/tensor
        self.masked_positions = list(masked_positions) if masked_positions is not None else []

    def __call__(self, examples):
        # pad via tokenizer 
        batch = self.tokenizer.pad(examples, return_tensors="pt")
        input_ids = batch["input_ids"]
        device = input_ids.device
        batch_size, seq_len = input_ids.shape

        # if no masked positions defined, return trivial labels
        if not self.masked_positions:
            typer.secho("[WARNING] no masked positions defined, returning trivial labels", fg=typer.colors.RED, bold=True)
            labels = input_ids.clone()
            batch["labels"] = labels
            return batch

        # masked positions tensor 
        pos_idx = torch.as_tensor(self.masked_positions, dtype=torch.long, device=device)

        # 1) compute counts at masked positions (0/1/2)
        masked_vals = input_ids.index_select(1, pos_idx)  # (batch, n_masked_pos)
        flat = masked_vals.reshape(-1)
        # keep only tokens in {0,1,2}
        valid_mask = (flat >= 0) & (flat <= 2)
        if valid_mask.any():
            counts = torch.bincount(flat[valid_mask].to(torch.long), minlength=3)
            target_0 = int(counts[0].item())
            target_1 = int(counts[1].item())
            target_2 = int(counts[2].item())
        else:
            target_0 = target_1 = target_2 = 0

        # 2) desired balanced targets (same logic as before)
        target_1_balanced = min(target_0, int(self.mult_1 * target_1))
        target_2_balanced = min(target_0, int(self.mult_2 * target_2))

        extra_1 = max(0, target_1_balanced - target_1)
        extra_2 = max(0, target_2_balanced - target_2)

        # distribute extras across samples (integer division)
        extra_1_per_sample = extra_1 // batch_size
        extra_2_per_sample = extra_2 // batch_size

        # 3) Prepare labels and always-mask positions
        labels = input_ids.clone()
        # always mask the masked_positions (use standard slicing for worker-safe indexing)
        # scalar assignment will broadcast to shape (batch_size, num_masked_pos)
        input_ids[:, pos_idx] = self.mask_token_id
        # loss mask: -100 everywhere except masked positions (and extras set below)
        loss_mask = torch.full_like(labels, -100)
        loss_mask[:, pos_idx] = labels.index_select(1, pos_idx)

        # 4) Candidate positions: positions not in masked_positions and not pad (use first row heuristic like original)
        mask_all_pos = torch.ones(seq_len, dtype=torch.bool, device=device)
        mask_all_pos[pos_idx] = False
        # use first sample to detect padding positions (columns)
        mask_all_pos &= (input_ids[0] != self.pad_token_id)
        candidate_idx = torch.nonzero(mask_all_pos, as_tuple=False).squeeze(1)
        if candidate_idx.numel() == 0 or (extra_1_per_sample == 0 and extra_2_per_sample == 0):
            batch["input_ids"] = input_ids
            batch["labels"] = loss_mask
            return batch

        # tokens at candidate positions: shape (batch, num_candidates)
        tokens_cand = input_ids.index_select(1, candidate_idx)

        # create weight matrices for tokens==1 and tokens==2
        weights1 = (tokens_cand == 1).to(torch.float32)
        weights2 = (tokens_cand == 2).to(torch.float32)

        # 5) For each sample, randomly pick up to extra_N_per_sample positions from available candidates
        # loop over batch 
        for i in range(batch_size):
            # pick for class 1
            if extra_1_per_sample > 0:
                w = weights1[i]
                avail = int(w.sum().item())
                if avail > 0:
                    n = min(extra_1_per_sample, avail)
                    if n > 0:
                        idx_in_cand = torch.multinomial(w, num_samples=n, replacement=False)
                        pos_to_mask = candidate_idx[idx_in_cand]
                        input_ids[i, pos_to_mask] = self.mask_token_id
                        loss_mask[i, pos_to_mask] = labels[i, pos_to_mask]

            # pick for class 2
            if extra_2_per_sample > 0:
                w = weights2[i]
                avail = int(w.sum().item())
                if avail > 0:
                    n = min(extra_2_per_sample, avail)
                    if n > 0:
                        idx_in_cand = torch.multinomial(w, num_samples=n, replacement=False)
                        pos_to_mask = candidate_idx[idx_in_cand]
                        input_ids[i, pos_to_mask] = self.mask_token_id
                        loss_mask[i, pos_to_mask] = labels[i, pos_to_mask]

        # put back into batch and return
        batch["input_ids"] = input_ids
        batch["labels"] = loss_mask
        return batch

# https://raw.githubusercontent.com/itakurah/Focal-loss-PyTorch/main/focal_loss.py
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean', task_type='multi-class', num_classes=None):
        """
        Unified Focal Loss class for binary, multi-class, and multi-label classification tasks.
        :param gamma: Focusing parameter, controls the strength of the modulating factor (1 - p_t)^gamma
        :param alpha: Balancing factor, can be a scalar or a tensor for class-wise weights. If None, no class balancing is used.
        :param reduction: Specifies the reduction method: 'none' | 'mean' | 'sum'
        :param task_type: Specifies the type of task: 'binary', 'multi-class', or 'multi-label'
        :param num_classes: Number of classes (only required for multi-class classification)
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.task_type = task_type
        self.num_classes = num_classes

        # Handle alpha for class balancing in multi-class tasks
        if task_type == 'multi-class' and alpha is not None and isinstance(alpha, (list, torch.Tensor)):
            assert num_classes is not None, "num_classes must be specified for multi-class classification"
            if isinstance(alpha, list):
                self.alpha = torch.Tensor(alpha)
            else:
                self.alpha = alpha

    def forward(self, inputs, targets):
        return self.multi_class_focal_loss(inputs, targets)

    def multi_class_focal_loss(self, inputs, targets):
        """ Focal loss for multi-class classification. """
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)

        # Convert logits to probabilities with softmax
        probs = F.softmax(inputs, dim=1)

        # One-hot encode the targets
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()

        # Compute cross-entropy for each class
        ce_loss = -targets_one_hot * torch.log(probs)

        # Compute focal weight
        p_t = torch.sum(probs * targets_one_hot, dim=1)  # p_t for each sample
        focal_weight = (1 - p_t) ** self.gamma

        # p_t = predicted prob for the true class
        # alpha_t = per-class weight (e.g [0.5, 1, 1.5, 0, 0, 0, 0, 0]
        # gamma = controls how much to focus on hard samples

        # Apply alpha if provided (per-class weighting)
        if self.alpha is not None:
            alpha_t = alpha.gather(0, targets)
            ce_loss = alpha_t.unsqueeze(1) * ce_loss

        # Apply focal loss weight
        loss = focal_weight.unsqueeze(1) * ce_loss

        if self.reduction == 'mean': # CHANGED, ignore where alpha==0
            mask = (alpha_t > 0).float()  # shape: (batch_size,)
            loss_per_sample = loss.sum(dim=1)  # shape: (batch_size,)
            return (loss_per_sample * mask).sum() / mask.sum()
            #return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# https://github.com/huggingface/transformers/blob/v4.52.3/src/transformers/models/bert/modeling_bert.py#L1250
class BertForMaskedLMWithFocalLoss(BertForMaskedLM):
    def __init__(self, config, alpha=[1.0] * 8, gamma=2, task_type='multi-class', num_classes=8, reduction='mean'):
        super().__init__(config)
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma, task_type=task_type, num_classes=num_classes, reduction=reduction)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            # flatten predictions: (batch_size * seq_len, vocab_size)
            logits = prediction_scores.view(-1, self.config.vocab_size)
            # flatten labels: (batch_size * seq_len)
            flat_labels = labels.view(-1)
            #print("labels unique values:", torch.unique(labels)) # {-100, 0, 1, 2}

            # mask out -100 entries
            valid_indices = flat_labels != -100
            logits = logits[valid_indices]
            flat_labels = flat_labels[valid_indices]

            # now compute focal loss only on valid tokens === LOSS ONLY ON [MASK] !
            masked_lm_loss = self.focal_loss(logits, flat_labels)

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
def compute_metrics(pred):
    logits = pred.predictions  # (batch_size, seq_len, vocab_size)
    labels = pred.label_ids    # (batch_size, seq_len)

    preds = np.argmax(logits, axis=-1)
    mask = labels != -100 # ignore -100 positions - pad/not masked

    masked_labels = labels[mask]
    masked_preds = preds[mask]

    accuracy = (masked_preds == masked_labels).sum() / len(masked_labels)
    f1 = f1_score(masked_labels, masked_preds, average='micro')
    f1_macro = f1_score(masked_labels, masked_preds, average='macro')

    # compute average cross-entropy per masked token in numpy (stable)
    active_logits = logits[mask]  # shape (N, vocab)
    active_labels = masked_labels  # shape (N,)

    # stable log-softmax
    max_logits = np.max(active_logits, axis=1, keepdims=True)
    shifted = active_logits - max_logits
    logsumexp = np.log(np.sum(np.exp(shifted), axis=1)) + max_logits.squeeze()
    log_probs = active_logits - logsumexp[:, None]
    neg_log_probs = -log_probs[np.arange(len(active_labels)), active_labels]
    cross_entropy_avg = neg_log_probs.mean()

    # keep small sample of per-token losses for logging
    per_token_sample = neg_log_probs[:3]
    per_token_sample_str = " ".join(f"{loss:.4f}" for loss in per_token_sample)

    typer.secho(
            f"Eval metrics: "
            f"acc={accuracy:.4f}  "
            f"f1_micro={f1:.4f}  "
            f"f1_macro={f1_macro:.4f}  "
            f"ce={cross_entropy_avg:.4f}\n",
            fg=typer.colors.GREEN, 
            bold=True
        )

    return {
        'accuracy': accuracy,
        'f1_micro': f1,
        'f1_macro': f1_macro,
        'cross_entropy_avg': float(cross_entropy_avg),
        'crossentropy_per_token': per_token_sample_str
    }

def build_model(hidden, layers, heads, vocab_size, max_position_embeddings, loss="focal", class_alpha=None, gamma=3,
                intermediate_size=256):
    config = BertConfig(vocab_size=vocab_size, hidden_size=hidden, num_hidden_layers=layers, num_attention_heads=heads,
                        max_position_embeddings=max_position_embeddings, intermediate_size=intermediate_size)
    if loss=="focal":
        model = BertForMaskedLMWithFocalLoss(config, alpha=class_alpha, gamma=gamma, num_classes=vocab_size)
    else:
        model = BertForMaskedLM(config)
    return model


@app.command()
def train(
    train_path: str = typer.Option(..., help="CSV path for training genotypes (no missing values)."),
    miss_path: str = typer.Option(None, help="Optional CSV path for masked genotypes (3 == missing)."),
    epochs: int = typer.Option(500),
    batch_size: int = typer.Option(128),
    sentence_size: int = typer.Option(512),
    heads: int = typer.Option(1, help="Number of attention heads- config param for BertForMaskedLM"),
    layers: int = typer.Option(2, help="Depth of the transformer - config param for BertForMaskedLM"),
    hidden: int = typer.Option(128, help = "Number of units in each layer - config param for BertForMaskedLM"),
    intermediate_size: int = typer.Option(1024, help="Feedforward inner dimension - config param for BertForMaskedLM"),
    gpu: bool = typer.Option(True, help="Whether to use GPU."),
    mult_1: float = typer.Option(1.0, help="Maximum oversampling for genotype 1 in mask."),
    mult_2: float = typer.Option(1.5, help="Maximum oversampling for genotype 2 in mask."),
    val_split: float = typer.Option(0.2, help="Validation split fraction."),
    loss: str = typer.Option("focal", help="Loss function (ce or focal)."),
    results_dir: str = typer.Option(None, help="Folder to save results if different from default."),
    gamma: float = typer.Option(1.45, help="Gamma for focal loss."),
    export_model_name: str = typer.Option("Bert", help="Model name for saving results."),
    lr: float = typer.Option(1e-3, help="Learning rate."),
):
    # -------------------------
    # Load data
    # -------------------------
    genotypes_train = read_csv_array(train_path)
    genotypes_test_missing = read_csv_array(miss_path) if miss_path else None

    if genotypes_train is None:
        raise typer.BadParameter("Missing required input file (--train-path).")
    
    typer.secho(f"[INFO] Train data shape: {genotypes_train.shape}", fg=typer.colors.BRIGHT_GREEN)

    # -------------------------
    # Prepare result directories
    # -------------------------
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_dir = Path(results_dir) if results_dir else Path("results") / f"model_BERT_{timestamp}_{export_model_name}"
    model_save_path = result_dir / "model"
    log_dir = result_dir / "logs"
    result_dir.mkdir(parents=True, exist_ok=True)
    #log_dir.mkdir(parents=True, exist_ok=True)
    model_save_path.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Prepare datasets
    # -------------------------
    processor = SNPDataProcessor(
        genotypes_train=genotypes_train,
        genotypes_test_missing=genotypes_test_missing,
        sentence_size=sentence_size,
        val_fraction=val_split
    )

    train_tokens, _, _ = processor.prepare_datasets(vocab_file=result_dir / "vocab.txt")
    train_dataset = DictDataset(train_tokens)
    train_dataset, eval_dataset = processor.split_dataset(train_dataset)

    typer.secho(f"[INFO] Train dataset: {len(train_dataset)}, Eval dataset: {len(eval_dataset)}", fg=typer.colors.BRIGHT_CYAN)

    collator = SNPColumnBalancingMaskingCollator(
        tokenizer=processor.tokenizer,
        masked_positions=processor.missing_cols_ids,
        mult_1=mult_1,
        mult_2=mult_2
    )

    # -------------------------
    # Compute inverse-frequency class weights
    # -------------------------
    class_alpha_list = None
    try:
        train_input_ids = train_tokens["input_ids"].numpy()
        masked_pos = processor.missing_cols_ids or []
        if len(masked_pos) > 0:
            # select tokens at masked positions and flatten
            vals = train_input_ids[:, masked_pos].reshape(-1)
            # keep only genotype tokens 0/1/2
            valid_mask = (vals >= 0) & (vals <= 2)
            if valid_mask.any():
                vals_valid = vals[valid_mask].astype(int)
                counts = np.bincount(vals_valid, minlength=3)
                total = counts.sum()
                inv_freqs = np.array([ (total / c) if c > 0 else 0.0 for c in counts ], dtype=float)
                # normalize so average weight ~ 1 (ignore zeros in mean)
                nonzero = inv_freqs[inv_freqs > 0]
                if nonzero.size > 0:
                    inv_freqs = inv_freqs / nonzero.mean()
                class_alpha_list = inv_freqs.tolist()
                typer.secho(f"[INFO] Computed class alpha (inverse frequency) = {class_alpha_list}", fg=typer.colors.BRIGHT_MAGENTA)
            if sum(valid_mask) == 0:
                print(vals)
                raise ValueError("Sum of valid_mask is zero, cannot continue.")
    except Exception as e:
        typer.secho(f"[ERROR] Could not compute class alpha automatically: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # -------------------------
    # Build model and choose device
    # -------------------------
    steps_per_epoch = genotypes_train.shape[0] // batch_size
    typer.secho(f"[INFO] Steps per epoch: {steps_per_epoch}", fg=typer.colors.BRIGHT_CYAN)
    model = build_model(
        hidden=hidden,
        layers=layers,
        heads=heads,
        vocab_size=len(processor.tokenizer),
        loss=loss,
        class_alpha=class_alpha_list,
        gamma=gamma,
        max_position_embeddings=sentence_size,
        intermediate_size=intermediate_size
    )

    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        typer.secho(f"[INFO] Using GPU: {props.name}, memory {props.total_memory / 1e9:.1f} GB", fg=typer.colors.BRIGHT_GREEN)
    else:
        typer.secho("[INFO] Training on CPU", fg=typer.colors.BRIGHT_YELLOW)

    # -------------------------
    # Setup training arguments
    # -------------------------
    use_fp16 = device.type == "cuda"
    eval_strategy = "steps" if len(eval_dataset) > 0 else "no"

    training_args = TrainingArguments(
        overwrite_output_dir=True,
        output_dir=str(result_dir),
        logging_dir=str(log_dir),
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=lr,
        logging_steps=steps_per_epoch, # log every epoch
        eval_strategy=eval_strategy,
        eval_steps=steps_per_epoch, # every epoch evaluate
        save_steps=steps_per_epoch*5,
        save_total_limit=2,
        dataloader_num_workers=4,
        dataloader_pin_memory=use_fp16,
        fp16=use_fp16,
        gradient_accumulation_steps=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        #report_to=[]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if len(eval_dataset) > 0 else None,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5),
                   EpochLoggerCallback()]
    )

    # svae to onnx
    dummy_input_ids = torch.randint(
        0,
        model.config.vocab_size,
        (batch_size, sentence_size),
        dtype=torch.long
    )
    dummy_attention_mask = torch.ones(
        (batch_size, sentence_size),
        dtype=torch.long
    )
    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask),
        "my_model.onnx",
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch", 1: "sequence"}
        },
        opset_version=14,
        do_constant_folding=True
    )

    # -------------------------
    # Train model
    # -------------------------
    start_time = time.perf_counter()
    trainer.train()
    end_time = time.perf_counter()
    train_time_sec = end_time - start_time
    train_time_min = train_time_sec / 60

    trainer.save_model(model_save_path)
    checkpoint_dirs = [d for d in Path(result_dir).glob("checkpoint-*") if d.is_dir()]
    for chk in checkpoint_dirs:
        shutil.rmtree(chk)
    processor.tokenizer.save_pretrained(str(model_save_path))
    history = trainer.state.log_history
    eval_logs = [x for x in history if "eval_loss" in x and "eval_f1_macro" in x]

    if eval_logs:
        # find the best val_loss
        best_entry = min(eval_logs, key=lambda x: x["eval_loss"])
        best_val_loss = best_entry["eval_loss"]
        f1_at_best_loss = best_entry["eval_f1_macro"]
        best_epoch = best_entry["epoch"]  # or 'epoch' if you log it separately

        csv_path = Path(result_dir) / "train_metrics.csv"
        with open(csv_path, "w") as f:
            f.write("model,model_path,val_loss_best,val_f1_macro,best_epoch\n")
            f.write(f"{export_model_name},{model_save_path},{best_val_loss},{f1_at_best_loss},{best_epoch}\n")

    else:
        print("No evaluation logs found.")

    # -------------------------
    # Save metadata 
    # -------------------------
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL INFO] params = {total_params} | trainable_params = {trainable_params}")
    params = {
        'timestamp': timestamp,
        'export_model_name': export_model_name,
        'sentence_size': sentence_size,
        'heads': heads,
        'layers': layers,
        'hidden': hidden,
        'intermediate_size': intermediate_size,
        'batch_size': batch_size,
        'epochs': epochs,
        'loss': loss,
        'learning_rate': lr,
        'class_alpha': class_alpha_list,
        "mult_1": mult_1,
        "mult_2": mult_2,
        "model_params": total_params,
        "model_trainable_params": trainable_params,
        "train_time_sec_total": float(train_time_sec),
        "train_time_min_total": float(train_time_min)
    }
    with open(result_dir / 'params.txt', 'w') as f:
        for k, v in params.items():
            f.write(f"{k}: {v}\n")

    typer.secho(f"[INFO] Training complete. Run artifacts saved to {result_dir}", fg=typer.colors.BRIGHT_GREEN)
    return str(result_dir)


@app.command()
def eval(
    test_path: str = typer.Option(..., help="CSV path for masked genotypes (3 == missing)"),
    full_path: str = typer.Option(..., help="CSV path for full genotypes (ground truth)"),
    model_dir: str = typer.Option(..., help="Path to trained model folder"),
    maf_path: str = typer.Option(None, help="[Optional] MAF scores for evaluation."),
    sentence_size: int = typer.Option(512),
    batch_size: int = typer.Option(64),
    eval_parent_dir: str = typer.Option(None, help="Optional parent directory to place eval outputs (e.g., training run dir). If provided, eval results will be written to <parent>/eval_<timestamp>."),
    export_model_name: str = typer.Option("Bert", help="Model name for saving in output csv results.")
):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = Path(eval_parent_dir) if eval_parent_dir else Path("results")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if maf_path is not None:
        data_maf = read_maf(maf_path)
    else:
        data_maf = None

    
    # read data
    genotypes_test_missing = read_csv_array(test_path)
    genotypes_test_full = read_csv_array(full_path)
    if genotypes_test_missing is None or genotypes_test_full is None:
        raise typer.BadParameter("Missing required input files (--test_path, --full_path).")

    # prepara data processor & tokenizer 
    processor = SNPDataProcessor(genotypes_test_missing=genotypes_test_missing,
                                 genotypes_test_full=genotypes_test_full,
                                 sentence_size=sentence_size)
    try:
        tokenizer = BertTokenizerFast.from_pretrained(model_dir)
    except Exception:
        # fallback: create a local tokenizer from `vocab.txt`
        tokenizer = None

    # process data
    test_sentences = processor.genotype_to_sentences(genotypes_test_missing)
    masked_sentences = [" ".join("[MASK]" if x == "3" else x for x in sent.split()) for sent in test_sentences]
    if tokenizer is None:
        tokenizer = BertTokenizerFast(vocab_file=str(Path(model_dir) / "vocab.txt"), do_lower_case=False)
    tokenized_test_masked = tokenizer(masked_sentences, padding=True, truncation=False, return_tensors="pt")
    tokenized_test_full = tokenizer(processor.genotype_to_sentences(genotypes_test_full), padding=True, truncation=False, return_tensors="pt")

    test_dataset = DictDataset({"input_ids": tokenized_test_masked["input_ids"], "labels": tokenized_test_full["input_ids"], "attention_mask": tokenized_test_masked["attention_mask"]})

    model = BertForMaskedLM.from_pretrained(model_dir)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # predict
    start_time = time.perf_counter()
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    all_preds = []
    all_labels = []
    all_masks = []  # store the full mask for all positions
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
        pred_token_ids = torch.argmax(probs, dim=-1)
        mask = input_ids == tokenizer.mask_token_id
        preds_filtered = pred_token_ids[mask].cpu().numpy()
        labels_filtered = labels[mask].cpu().numpy()
        all_preds.extend(preds_filtered.tolist())
        all_labels.extend(labels_filtered.tolist())
        all_masks.append(mask.cpu().numpy())
    end_time = time.perf_counter()
    eval_time_sec_total = end_time - start_time
    eval_time_min_total = eval_time_sec_total / 60

    total_params = sum(p.numel() for p in model.parameters())
    #trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    acc = accuracy_score(all_labels, all_preds)
    f1_micro = f1_score(all_labels, all_preds, average="micro")
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    f1_per_class = f1_score(all_labels, all_preds, labels=[0,1,2], average=None)
    typer.echo(typer.style(f"Model params: {total_params}", fg=typer.colors.BRIGHT_CYAN, bold=True))
    typer.echo(typer.style(f"Accuracy: {acc:.4f}", fg=typer.colors.BRIGHT_CYAN, bold=True))
    typer.echo(typer.style(f"F1 (micro): {f1_micro:.4f}", fg=typer.colors.BRIGHT_CYAN, bold=True))
    typer.echo(typer.style(f"F1 (macro): {f1_macro:.4f}", fg=typer.colors.BRIGHT_CYAN, bold=True))
    typer.echo(typer.style(f"F1 per class [0,1,2]: {f1_per_class}", fg=typer.colors.BRIGHT_CYAN, bold=True))
    report = classification_report(all_labels, all_preds)
    typer.echo(report)

    if data_maf is not None:
            csv_pathMAF = save_dir / "MAF_snp_eval_metrics.csv"
            file_existsMAF = csv_pathMAF.exists()

            metrics = maf_stratified_metrics(
                all_labels, all_preds, data_maf, all_masks
            )[1]
            typer.secho("[EVAL] Classification Report for rare SNPs", fg=typer.colors.BRIGHT_MAGENTA)
            #print("\nClassification Report for rare SNPs:")
            print(f"Description: {metrics['desc']}")
            
            print("Report:\n")
            print(metrics['report'])
            
            print(f"F1 (micro): {metrics['f1_micro']:.4f}")
            print(f"F1 (macro): {metrics['f1_macro']:.4f}\n")
            print(f"F1 per class [0,1,2]: {metrics['f1_per_class']}")
            
            print("Confusion Matrix:")
            print(metrics['confusion_matrix'])

            with open(csv_pathMAF, "a", newline="") as f:
                writer = csv.writer(f)
                if not file_existsMAF:
                    writer.writerow(
                        ["model", "model_path", "date", "dataset", "f1_micro", "f1_macro", "f1_0", "f1_1", "f1_2", "n_params"]
                    )
                writer.writerow([
                    str(export_model_name), str(model_dir), timestamp, test_path, metrics['f1_micro'], metrics['f1_macro'], metrics['f1_per_class'][0], metrics['f1_per_class'][1], metrics['f1_per_class'][2], total_params
                ])

            typer.secho(f"[INFO] MAF evaluation report saved to '{csv_pathMAF}'", fg=typer.colors.GREEN)

    # --- save simplified CSV ---
    csv_path = save_dir / "snp_eval_metrics.csv"
    file_exists = csv_path.exists()
    row = [str(export_model_name), str(model_dir), timestamp, test_path, f1_micro, f1_macro,
        f1_per_class[0], f1_per_class[1], f1_per_class[2], total_params, eval_time_sec_total, eval_time_min_total]

    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["model", "model_path", "date", "dataset", "f1_micro", "f1_macro", "f1_0", "f1_1", "f1_2", "n_params", "eval_time_sec_total", "eval_time_min_total"])
        writer.writerow(row)

    typer.secho(f"Saved simplified metrics to {csv_path}", fg=typer.colors.BRIGHT_GREEN)


@app.command("train-and-eval")
def train_and_eval(
    train_path: str = typer.Option(..., help="CSV path for training genotypes (no missing values)."),
    test_path: str = typer.Option(..., help="CSV path for masked test genotypes (3 == missing)."),
    test_full_path: str = typer.Option(..., help="CSV path for full test genotypes (ground truth)."),
    sentence_size: int = typer.Option(512),
    maf_path: str = typer.Option(None, help="[Optional] MAF scores for evaluation."),
    heads: int = typer.Option(1, help="Number of attention heads- config param for BertForMaskedLM"),
    layers: int = typer.Option(2, help="Depth of the transformer - config param for BertForMaskedLM"),
    hidden: int = typer.Option(128, help = "Number of units in each layer - config param for BertForMaskedLM"),
    intermediate_size: int = typer.Option(1024, help="Feedforward inner dimension - config param for BertForMaskedLM. Up to 3072."),
    val_split: float = typer.Option(0.2, help="Validation split"),
    mult_1: float = typer.Option(1.0, help="Maximum oversampling for genotype 1 in mask."),
    mult_2: float = typer.Option(1.5, help="Maximum oversampling for genotype 2 in mask."),
    batch_size: int = typer.Option(128),
    epochs: int = typer.Option(500),
    loss: str = typer.Option("focal", help="Loss function (ce or focal)"),
    gamma: float = typer.Option(1.45, help="Gamma for focal loss"),
    results_dir: str = typer.Option(None, help="Folder to save result to if different than results/."),
    gpu: bool = typer.Option(True, help="Whether to use GPU (default True). If False, forces CPU training."),
    export_model_name: str = typer.Option("Bert", help="Model name for saving in output csv results."),
    lr: float = typer.Option(1e-3, help="Learning rate."),
):
    typer.secho("[INFO] Running train then eval (train-and-eval)", fg=typer.colors.BRIGHT_CYAN)

    run_dir = train(
        train_path=train_path,
        miss_path=test_path,
        sentence_size=sentence_size,
        heads=heads,
        layers=layers,
        hidden=hidden,
        batch_size=batch_size,
        epochs=epochs,
        loss=loss,
        gamma=gamma,
        gpu=gpu,
        mult_1=mult_1,
        mult_2=mult_2,
        export_model_name=export_model_name,
        results_dir=results_dir,
        val_split=val_split,
        intermediate_size=intermediate_size,
        lr=lr
    )

    model_path_in_run = Path(run_dir) / "model"

    typer.secho(f"[INFO] Running evaluation using model at {model_path_in_run}", fg=typer.colors.BRIGHT_CYAN)
    eval(test_path=test_path, full_path=test_full_path, sentence_size=sentence_size, batch_size=batch_size,
         export_model_name=export_model_name, model_dir=str(model_path_in_run), 
         eval_parent_dir=str(run_dir), maf_path=maf_path)


if __name__ == "__main__":
    typer.secho(f"[INFO] PyTorch {torch.__version__} | CUDA available: {torch.cuda.is_available()} | GPUs: {torch.cuda.device_count()}", fg=typer.colors.BRIGHT_GREEN)
    app()