#!/bin/sh

for i in $(seq 0 4)
do
  python main_sda_dis.py 2018 $i 2021 CVL > log_sda_dis_cvl_2018_2021_$i
  python main_sda_dis.py 2021 $i 2018 CVL > log_sda_dis_cvl_2021_2018_$i
done
