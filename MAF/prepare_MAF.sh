#!/bin/bash

# set input VCF file
VCF_FILE="baseline/TestFull.vcf"

# set output prefix
OUT_PREFIX="TestFull"

# convert VCF to PLINK binary format
plink2 --vcf $VCF_FILE --make-bed --out $OUT_PREFIX --chr-set 29

# calculate minor allele frequencies
plink2 --bfile $OUT_PREFIX --freq --out "${OUT_PREFIX}_maf" --chr-set 29

echo "PLINK conversion and MAF calculation finished. Output files: ${OUT_PREFIX}.bed/.bim/.fam and ${OUT_PREFIX}_maf.frq"
