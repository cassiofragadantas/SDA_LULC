#!/bin/sh

for i in $(seq 0 4)
do
  python rf_direct.py 2018 $i 2021 > log_direct_transfer_rf_2018_2021_$i
  python rf_direct.py 2021 $i 2018 > log_direct_transfer_rf_2021_2018_$i
  python rf_direct.py 2020 $i 2018 > log_direct_transfer_rf_2020_2018_$i
  python rf_direct.py 2018 $i 2020 > log_direct_transfer_rf_2018_2020_$i
  python rf_direct.py 2020 $i 2021 > log_direct_transfer_rf_2020_2021_$i
  python rf_direct.py 2021 $i 2020 > log_direct_transfer_rf_2021_2020_$i
done