#!/bin/bash

beta=$1
epochs=$2
alpha=$3

dataset="covid"
time python3 WTM/WTM_run.py -n $dataset -l "$dataset"_metadata.csv -f sample_data/"$dataset"_data -e "$dataset"_gene.csv,"$dataset"_microbeR.csv -m "gene,microbeR" --n_topic 10 --num_epochs ${epochs} --learning_rate 0.0005 --dir_alpha ${alpha} --beta ${beta}
