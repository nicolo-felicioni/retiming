from GraphWrapper import *
import time
from random import randint
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pandas as pd
import cProfile
import io
import pstats


def draw():

    g = nx.DiGraph(nx.nx_pydot.read_dot("graph_files/bug5n9e"))
    nx.draw_networkx(g)
    # print(type(g))
    plt.show()

    # convert node labels to int
    g = nx.relabel_nodes(g, lambda x: int(x))


    # convert node delays to float
    for node in g.nodes:
        g.nodes[node]["delay"] = float(g.nodes[node]["delay"])


    # convert edge w to int
    for e in g.edges:
        g.edges[e]["weight"] = int(g.edges[e]["weight"])

    wrapper = GraphWrapper(g)
    wrapper.init_WD()

    print(wrapper.opt2())


def main():

    g = nx.binomial_graph(n=100, p=0.05, seed=42, directed=True)

    for edge in g.edges:
        g.edges[edge]["weight"] = randint(1, 5)

    for node in g.nodes:
        g.nodes[node]["delay"] = randint(1, 10)

    print(f"Optimal CP: {max([g.nodes[node]['delay'] for node in g.nodes])}")

    wrapper = GraphWrapper(g)

    r = random_retime(wrapper.g)

    wrapper.set_retimed_graph(r)
    print("legal retiming found.")

    _, cp = cp_delta_clock(wrapper.g)
    wrapper.init_WD()

    print(f'previous clock period: {cp}')
    print(f'unique val of D: {len(np.unique(wrapper.D))}')
    print(f'num of nodes: {len(g.nodes)}')
    print(f'num of edges: {len(g.edges)}')

    t0 = time.time()
    print(wrapper.opt1())
    t1 = time.time()
    print(wrapper.opt2())
    t2 = time.time()
    print(wrapper.opt2_optimized())
    t3 = time.time()

    print(f"opt1:{t1 - t0}")
    print(f"opt2:{t2 - t1}")
    print(f"opt2_optimized:{t3 - t2}")
    return g, wrapper


def generate_test():

    for N in tqdm([5, 10, 20, 50, 100, 200, 500]):
        for p in tqdm([.0, .05, .1, .2, .3, .5, .75, 1]):
            for up_w in tqdm([1, 5, 10, 100, 1000, 10000]):
                for up_d in tqdm([1, 5, 10, 100, 1000, 10000]):
                    # generate a cyclic backbone
                    backbone_graph = nx.DiGraph()
                    for n in range(N):
                        backbone_graph.add_edge(n, (n + 1) % N)

                    random_graph = nx.binomial_graph(n=N, p=p, seed=42, directed=True)

                    graph = nx.compose(backbone_graph, random_graph)

                    for edge in graph.edges:
                        graph.edges[edge]["weight"] = randint(1, up_w)

                    for node in graph.nodes:
                        graph.nodes[node]["delay"] = randint(1, up_d)

                    wrapper = GraphWrapper(graph)

                    r = random_retime(graph)

                    # randomization of the graph
                    wrapper.set_retimed_graph(r)

                    nx.nx_pydot.write_dot(wrapper.g, f"graph_files/N_{N}_p_{p}_upw_{up_w}_upd_{up_d}")


def test(path):
    print("-"*30)
    print(f"start test of {path}")
    print("-"*30)

    test_graph = read_graph_dot(path)
    optimal_cp = max([test_graph.nodes[node]['delay'] for node in test_graph.nodes])
    print(f"Optimal CP: {optimal_cp}")

    wrapper = GraphWrapper(test_graph)

    wrapper.init_WD()

    print(f'unique val of D: {len(np.unique(wrapper.D))}')
    print(f'num of nodes: {len(test_graph.nodes)}')
    print(f'num of edges: {len(test_graph.edges)}')

    t0 = time.time()
    cp1, _ = wrapper.opt1()
    t1 = time.time()
    cp2, _ = wrapper.opt2()
    t2 = time.time()
    cp2_optimized, _ = wrapper.opt2_optimized()
    t3 = time.time()

    print(f"opt1:{t1 - t0}")
    print(f"opt2:{t2 - t1}")
    print(f"opt2_optimized:{t3 - t2}")

    assert cp1 == cp2 == cp2_optimized == optimal_cp, f"something went wrong." \
                                                      f"optimal: {optimal_cp}" \
                                                      f"cp1: {cp1}, cp2: {cp2}, cp2_optimized: {cp2_optimized}"
    print("tests ok")


def test_including_WD(path):
    print("-"*30)
    print(f"start test of {path}")
    print("-"*30)

    test_graph = read_graph_dot(path)
    optimal_cp = max([test_graph.nodes[node]['delay'] for node in test_graph.nodes])
    print(f"Optimal CP: {optimal_cp}")

    wrapper = GraphWrapper(test_graph)

    # wrapper.init_WD()

    print(f'unique val of D: {len(np.unique(wrapper.D))}')
    print(f'num of nodes: {len(test_graph.nodes)}')
    print(f'num of edges: {len(test_graph.edges)}')

    # t0 = time.time()
    # cp1, _ = wrapper.opt1()
    # wrapper.W = None
    t1 = time.time()
    cp2, _ = wrapper.opt2()
    # wrapper.W = None
    t2 = time.time()
    # cp2_optimized, _ = wrapper.opt2_optimized()
    # t3 = time.time()

    # print(f"opt1:{t1 - t0}")
    print(f"opt2:{t2 - t1}")
    # print(f"opt2_optimized:{t3 - t2}")

    # assert cp1 == cp2 == cp2_optimized == optimal_cp, f"something went wrong." \
    #                                                   f"optimal: {optimal_cp}" \
    #                                                   f"cp1: {cp1}, cp2: {cp2}, cp2_optimized: {cp2_optimized}"
    print("tests ok")


def test_sorted(path, algo_num):

    assert algo_num == 1 or algo_num == 2, f"hey there's no OPT-{algo_num}."

    print("-"*30)
    print(f"START TEST of {path}")
    print("-"*30)

    test_graph = read_graph_dot(path)

    wrapper = GraphWrapper(test_graph)

    print("initializing WD...")
    t_init = time.time()
    wrapper.init_WD()
    t_wd = time.time()-t_init
    print(f'WD init time:{t_wd}')

    optimal_cp, _, _, _, _ = get_stats(wrapper)

    t_start_sort = time.time()
    print(f"opt{algo_num}: sorting D...")
    d_elems_sorted = np.unique(wrapper.D)
    t_sort = time.time() - t_start_sort
    print(f"sorted D in {t_sort}")

    if algo_num == 1:
        t0 = time.time()
        cp1, _ = wrapper.binary_search_minimum_bf(d_elems_sorted)
        t1 = time.time()
    else:
        t0 = time.time()
        cp1, _ = wrapper.binary_search_minimum_feas_optimized(d_elems_sorted)
        t1 = time.time()

    t_opt = t1 - t0
    print(f"opt{algo_num}:{t1 - t0}")

    print(f"total time: {t_wd+t_sort+t_opt}")

    assert cp1 == optimal_cp, f"something went wrong." \
                                                      f"optimal: {optimal_cp}" \
                                                      f"cp1: {cp1}"
    print("tests ok")



def multiple_tests():
    main_dir = "graph_files"
    file_list = os.listdir(main_dir)
    new_list = []
    for f in file_list:
        if "N_5_" in f:
            new_list.append(f)

    for f in file_list:
        test(os.path.join(main_dir, f))


def multiple_tests_sorted():
    main_dir = "graph_files"
    file_list = os.listdir(main_dir)
    new_list = []
    for f in file_list:
        if "N_5_" in f:
            new_list.append(f)

    for f in file_list:
        test_sorted(os.path.join(main_dir, f), 1)




if __name__ == '__main__':
    #g, wrapper = main()
    # draw()
    # generate_test()
    # pr = cProfile.Profile()
    # pr.enable()
    test_sorted(os.path.join("misc", "temp.dot"), 2)
    # pr.disable()
    # s = io.StringIO()
    # sortby = 'cumulative'
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats()
    # print(s.getvalue())
