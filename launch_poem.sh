#!/bin/sh

#$1: first year
#$2: second year
#$3: dataset

for i in $(seq 0 4)
do
  python poem.py $1 $i $2 $3 > log_poem_$1_$2_$3_$i
  python poem.py $2 $i $1 $3 > log_poem_$2_$1_$3_$i
done
