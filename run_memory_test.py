from Wrappers.GraphWrapper import *
import time
from random import randint
from memory_profiler import memory_usage
from Wrappers.NewGraphWrapper import NewGraphWrapper
import argparse
import os

path = None
algo_num = None

def get_row_from_tuple(tup: tuple):
    row = '\n'
    row += str(tup[0])
    for i in range(1, len(tup)):
        row += ','
        if isinstance(tup[i] , float):
            row += f"{tup[i]:.5f}"
        else:
            row += str(tup[i])

    return row


def test_get_stats():
    """
    :param path: the path of the graph file
    :param algo_num: 1 --> opt1, 2 --> opt2 old wrap, 3 --> opt2 new wrap
    :return:
    """

    assert algo_num == 1 or algo_num == 2 or algo_num == 3, f"hey there's no OPT-{algo_num}."

    print("-"*30)
    print(f"START TEST of {path}")
    print("-"*30)

    test_graph = read_graph_dot(path)

    if algo_num != 3:
        wrapper = GraphWrapper(test_graph)
    else:
        wrapper = NewGraphWrapper(test_graph)

    print("initializing WD...")
    t_init = time.time()
    wrapper.init_WD()
    t_wd = time.time()-t_init
    print(f'WD init time:{t_wd}')

    optimal_cp, n_unique_D, n_nodes, n_edges, n_edges_zero = get_stats(wrapper)

    t_start_sort = time.time()
    print(f"opt{algo_num}: sorting D...")
    d_elems_sorted = np.unique(wrapper.D)
    t_sort = time.time() - t_start_sort
    print(f"sorted D in {t_sort}")

    if algo_num == 1:
        t0 = time.time()
        cp1, _ = wrapper.binary_search_minimum_bf(d_elems_sorted)
        t1 = time.time()
    elif algo_num == 2:
        t0 = time.time()
        cp1, _ = wrapper.binary_search_minimum_feas_optimized(d_elems_sorted)
        t1 = time.time()
    else:
        # NEW WRAPPER
        t0 = time.time()
        cp1, _ = wrapper.binary_search_minimum_feas(d_elems_sorted)
        t1 = time.time()

    t_opt = t1 - t0
    if algo_num == 3:
        print(f"opt2 NEW WRAP:{t1 - t0}")
    else:
        print(f"opt{algo_num}:{t1 - t0}")

    print(f"total time: {t_wd+t_sort+t_opt}")

    is_ok = cp1 == optimal_cp
    assert is_ok, f"something went wrong.  optimal: {optimal_cp}" \
                                          f"cp1: {cp1}"
    print("tests ok")

    filename = os.path.basename(path)

    return filename, algo_num, n_nodes, n_edges, n_edges_zero, n_unique_D, is_ok


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path')
    parser.add_argument('--algo_num')
    parser.add_argument('--write')
    args = parser.parse_args()
    path = args.path
    algo_num = int(args.algo_num)

    mem, stats_tuple = memory_usage(test_get_stats, retval=True)
    mem = max(mem)
    stats_tuple = stats_tuple + (mem, )

    row = get_row_from_tuple(stats_tuple)
    if int(args.write) == 1:
        with open('Test/stat_files/test_memory.csv', 'a') as fd:
            fd.write(row)

