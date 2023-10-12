#!/bin/sh

for i in $(seq 0 4)
do
  python main_contra_sda.py 2018 $i 2021 > log_sda_2018_2021_$i
  python main_contra_sda.py 2021 $i 2018 > log_sda_2021_2018_$i
  python main_contra_sda.py 2018 $i 2020 > log_sda_2018_2020_$i
  python main_contra_sda.py 2020 $i 2018 > log_sda_2020_2018_$i
  python main_contra_sda.py 2020 $i 2021 > log_sda_2020_2021_$i
  python main_contra_sda.py 2021 $i 2020 > log_sda_2021_2020_$i
done
