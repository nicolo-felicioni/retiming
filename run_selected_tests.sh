#!/bin/sh
for algo_num in {1..3}; do
    for path in selected_tests/*; do
      python3 run_test.py --path $path --algo_num $algo_num --write 0
    done
done

