#! /bin/bash

X=$1
Y=$2
LOSS=$3

echo "$X^2 + $Y^2" | bc -l > $LOSS
