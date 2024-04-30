#!/bin/sh

for i in $(seq 0 4)
do
  python main_sourcerer.py 2020 $i 2021 Koumbia > log_sourcerer_koumbia_2020_2021_$i
  python main_sourcerer.py 2021 $i 2020 Koumbia > log_sourcerer_koumbia_2021_2020_$i
done
