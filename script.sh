#!/bin/sh
NAME="test_opt_new_get_retime_graph.profile"
python3 -m cProfile -o profiles/$NAME test_generator.py
snakeviz profiles/$NAME