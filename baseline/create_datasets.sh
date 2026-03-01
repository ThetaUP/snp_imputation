#!/bin/bash

# -------------------------------
# Parameters
# -------------------------------
input_vcf="data/Chr28-Run9-TAU-filter-HOL-RS-annotated-filtered-TEST.vcf"
input_vcf_train="data/Chr28-Run9-TAU-filter-HOL-RS-annotated-filtered-TRAIN.vcf"
n_variants=500
missing_fractions=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8)
ne=250 #5000 # Beagle effective pop. param

# -------------------------------
# Step 1: extract first n variants
# -------------------------------
echo "Creating subset VCF with first $n_variants variants..."
bcftools head -n $n_variants $input_vcf > TestFull.vcf
bcftools head -n $n_variants $input_vcf_train > TrainFull.vcf
echo "Subset VCF(test) saved: TestFull.vcf"
echo "Subset VCF(train) saved: TrainFull.vcf"

# -------------------------------
# Step 2: create CSV for full dataset
# -------------------------------
echo "Creating CSV for full dataset..."
csv_full="TestFull.csv"
bcftools query -f '[%GT\t]\n' TestFull.vcf \
| sed 's/0\/0/0/g; s/0\/1/1/g; s/1\/0/1/g; s/1\/1/2/g; s/0|0/0/g; s/0|1/1/g; s/1|0/1/g; s/1|1/2/g; s/\.\/\./,/g; s/\.|\.//g' \
| tr '\t' ',' > $csv_full

Rscript -e "
gt <- read.csv('$csv_full', header=FALSE, stringsAsFactors=FALSE)
gt_t <- t(gt)
gt_t <- gt_t[-nrow(gt_t), ]
write.table(gt_t, file='$csv_full', sep=',', row.names=FALSE, col.names=FALSE)
"

echo "CSV saved: $csv_full"

echo "Creating CSV for full dataset..."
csv_full="TrainFull.csv"
bcftools query -f '[%GT\t]\n' TrainFull.vcf \
| sed 's/0\/0/0/g; s/0\/1/1/g; s/1\/0/1/g; s/1\/1/2/g; s/0|0/0/g; s/0|1/1/g; s/1|0/1/g; s/1|1/2/g; s/\.\/\./,/g; s/\.|\.//g' \
| tr '\t' ',' > $csv_full

Rscript -e "
gt <- read.csv('$csv_full', header=FALSE, stringsAsFactors=FALSE)
gt_t <- t(gt)
gt_t <- gt_t[-nrow(gt_t), ]
write.table(gt_t, file='$csv_full', sep=',', row.names=FALSE, col.names=FALSE)
"

echo "CSV saved: $csv_full"

# -------------------------------
# Step 3: create missing datasets
# -------------------------------
for f in "${missing_fractions[@]}"; do
    pct=$(echo "$f*100" | bc)
    vcf_missing="TestMiss_${pct}pct.vcf"
    csv_missing="TestMiss_${pct}pct.csv"

    echo "Creating $pct% missing dataset..."

    # use Python to create random missing genotypes (whole missing SNP possitions aka columns in csv later on)
    python3 - <<EOF
import vcfpy
import random
random.seed(222)

infile = "TestFull.vcf"
outfile = "$vcf_missing"
missing_frac = $f

reader = vcfpy.Reader.from_path(infile)
writer = vcfpy.Writer.from_path(outfile, reader.header)

for record in reader:
    if random.random() < missing_frac:
        # set all genotypes for this SNP to missing
        for call in record.calls:
            call.data['GT'] = '.|.'
    writer.write_record(record)

writer.close()
EOF

    echo "VCF with $pct% missing saved: $vcf_missing"

    # run Beagle
    echo "Run Beagle baseline..."
    mkdir -p beagle
    beagle gt=$vcf_missing \
           out=beagle/outBeagle_${vcf_missing%.vcf} \
           ne=$ne \
           seed=222 \
           ref=TrainFull.vcf
    gunzip -f beagle/outBeagle_${vcf_missing%.vcf}.vcf.gz

    # create corresponding CSVs
    bcftools query -f '[%GT\t]\n' $vcf_missing \
    | sed 's/0\/0/0/g; s/0\/1/1/g; s/1\/0/1/g; s/1\/1/2/g; s/0|0/0/g; s/0|1/1/g; s/1|0/1/g; s/1|1/2/g; s/\.\/\./3/g; s/\.|\. /3/g' \
    | tr '\t' ',' > $csv_missing

    bcftools query -f '[%GT\t]\n' beagle/outBeagle_${vcf_missing} \
    | sed 's/0\/0/0/g; s/0\/1/1/g; s/1\/0/1/g; s/1\/1/2/g; s/0|0/0/g; s/0|1/1/g; s/1|0/1/g; s/1|1/2/g; s/\.\/\./3/g; s/\.|\. /3/g' \
    | tr '\t' ',' > beagle/outBeagle_${csv_missing}

    # transpose CSVs
    Rscript -e "
    gt <- read.csv('$csv_missing', header=FALSE, stringsAsFactors=FALSE)
    gt_t <- t(gt)
    gt_t <- gt_t[-nrow(gt_t), ]
    write.table(gt_t, file='$csv_missing', sep=',', row.names=FALSE, col.names=FALSE)
    "

    Rscript -e "
    gt <- read.csv('beagle/outBeagle_${csv_missing}', header=FALSE, stringsAsFactors=FALSE)
    gt_t <- t(gt)
    gt_t <- gt_t[-nrow(gt_t), ]
    write.table(gt_t, file='beagle/outBeagle_${csv_missing}', sep=',', row.names=FALSE, col.names=FALSE)
    "

    echo "CSV for $pct% missing saved and transposed: $csv_missing"
done

echo "All datasets created successfully."
