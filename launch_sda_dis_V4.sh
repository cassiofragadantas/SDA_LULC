#!/bin/sh
DATASET=Koumbia # The other option is CVL3, with years 2018 and 2021
YEAR1=2020
YEAR2=2021
for i in $(seq 0 4)
do
  python main_REFeD.py $YEAR1 $i $YEAR2 $DATASET > log_REFeD_$YEAR1_$YEAR2_$i
  python main_REFeD.py $YEAR2 $i $YEAR1 $DATASET > log_REFeD_$YEAR2_$YEAR1_$i
done
