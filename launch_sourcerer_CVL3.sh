#!/bin/sh

for i in $(seq 0 4)
do
  python main_sourcerer.py 2018 $i 2021 CVL3 > log_sourcerer_cvl3_2018_2021_$i
  python main_sourcerer.py 2021 $i 2018 CVL3 > log_sourcerer_cvl3_2021_2018_$i
done
