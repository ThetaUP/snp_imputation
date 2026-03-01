 # SNP imputation

DL-based methods for SNP imputation task.

Input files:
    - Training genotypes: rows are individuals, columns are SNPs, values are 0/1/2.
    - Masked test genotypes: value `3` for missing positions.
    - Full test genotypes: ground-truth values (0/1/2).

 - `train`: train the model on complete genotypes.
 - `eval`: evaluate a trained model on masked test data (3 = missing) against ground truth.
 - `train-and-eval`: train and then evaluate.


## 1. Convolutional AutoEncoder (CAE)

```powershell
 python CAE_imputation.py --help
 python CAE_imputation.py train-and-eval --help
 ```

## 2. BERT

To be added