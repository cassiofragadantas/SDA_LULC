#!/bin/sh

for i in $(seq 0 4)
do
  python rf_target.py $i 2021 > log_target_rf_2021_$i
  python rf_target.py $i 2018 > log_target_rf_2018_$i
  python rf_target.py $i 2020 > log_target_rf_2020_$i
done