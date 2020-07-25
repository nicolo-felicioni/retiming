#!/bin/sh
NAME="N_100_p_0d75_upw_1000_upd_1000_opt2_cython.profile"
python3 -m cProfile -o profiles/$NAME test_generator.py
snakeviz profiles/$NAME