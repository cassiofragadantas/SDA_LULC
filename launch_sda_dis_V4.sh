#!/bin/sh

for i in $(seq 0 4)
do
  python main_combined_source_target_dis_V4.py 2018 $i 2021 Koumbia > log_sda_dis_2018_2021_$i
  python main_combined_source_target_dis_V4.py 2021 $i 2018 Koumbia > log_sda_dis_2021_2018_$i
  python main_combined_source_target_dis_V4.py 2018 $i 2020 Koumbia > log_sda_dis_2018_2020_$i
  python main_combined_source_target_dis_V4.py 2020 $i 2018 Koumbia > log_sda_dis_2020_2018_$i
  python main_combined_source_target_dis_V4.py 2020 $i 2021 Koumbia > log_sda_dis_2020_2021_$i
  python main_combined_source_target_dis_V4.py 2021 $i 2020 Koumbia > log_sda_dis_2021_2020_$i
done
