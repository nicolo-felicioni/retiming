#!/bin/sh
counter=0
for algo_num in 1 3; do
    echo 'algo_num is' $algo_num
    for path in selected_tests/*; do
      echo $counter
      python3 run_test.py --path $path --algo_num $algo_num --write 1
      counter=$((counter+1))
    done
done

