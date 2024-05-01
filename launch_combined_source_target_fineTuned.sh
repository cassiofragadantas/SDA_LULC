#!/bin/sh

#$1: first year
#$2: second year
#$3: dataset

for i in $(seq 0 4)
do
  python main_combined_source_target_fineTune.py $1 $i $2 $3 > log_combined_target_source_fineTuned_$1_$2_$i
  python main_combined_source_target_fineTune.py $2 $i $1 $3 > log_combined_target_source_fineTuned_$2_$1_$i
done
