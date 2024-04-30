#!/bin/sh

for i in $(seq 0 4)
do
  python main_combined_source_target_dis_V4.py 2018 $i 2021 CVL3 > log_sda_dis_V4_cvl3_2018_2021_$i
  python main_combined_source_target_dis_V4.py 2021 $i 2018 CVL3 > log_sda_dis_V4_cvl3_2021_2018_$i
done
