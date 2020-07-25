#!/bin/sh
python3 -m cProfile -o profiles/N_200_p_0d75_upw_1000_upd_1000_algo3.profile run_test.py --path graph_files/N_200_p_0d75_upw_1000_upd_1000 --algo_num 3 --write 0
snakeviz profiles/N_200_p_0d75_upw_1000_upd_1000_algo3.profile