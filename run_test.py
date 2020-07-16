from GraphWrapper import *
import time
from random import randint


def main():

    g = nx.binomial_graph(n=200, p=0.25, seed=42, directed=True)

    for edge in g.edges:
        g.edges[edge]["weight"] = 1

    for node in g.nodes:
        g.nodes[node]["delay"] = 10

    wrapper = GraphWrapper(g)

    found_retiming = False

    while not found_retiming:
        try:
            # calculate a random legal retiming
            r = {}
            # for each node
            for node in wrapper.g.nodes:
                min_in_w_r = np.inf
                min_out_w_r = np.inf
                # for each incoming edge
                for e in wrapper.g.in_edges(node):
                    tail_node = e[0]
                    # calculate the temporary w of the retimed edge
                    # for the retiming of the tail node: if it is not set, take 0
                    w_r = wrapper.g.edges[e]["weight"] - r.get(tail_node, 0)
                    assert w_r >= 0, "the retiming is not legal."

                    # take note of what is the minimum retimed incoming weight
                    if w_r < min_in_w_r:
                        min_in_w_r = w_r

                # for each outcoming edge
                for e in wrapper.g.out_edges(node):
                    head_node = e[1]
                    # calculate the temporary w of the retimed edge
                    # for the retiming of the tail node: if it is not set, take 0
                    w_r = wrapper.g.edges[e]["weight"] + r.get(head_node, 0)
                    assert w_r >= 0, "the retiming is not legal."

                    # take note of what is the minimum retimed incoming weight
                    if w_r < min_out_w_r:
                        min_out_w_r = w_r

                # now we have lowerbound and upperbound for the random choice
                r[int(node)] = randint(-min_in_w_r, min_out_w_r)

            wrapper.set_retimed_graph(r)
            found_retiming = True
        except ValueError:
            print("whoops.")
            found_retiming = False

    print("legal retiming found.")

    _, cp = cp_delta_clock(wrapper.g)
    wrapper.init_WD()

    print(f'previous clock period: {cp}')
    print(f'unique val of D: {len(np.unique(wrapper.D))}')
    print(f'num of nodes: {len(g.nodes)}')
    print(f'num of edges: {len(g.edges)}')

    # t0 = time.time()
    # print(wrapper.opt1())
    t1 = time.time()
    print(wrapper.opt2())
    t2 = time.time()

    # print(f"opt1:{t1 - t0}")
    print(f"opt2:{t2 - t1}")
    return wrapper

def test_initialized():

    g = nx.binomial_graph(n=150, p=0.25, seed=42, directed=True)

    for edge in g.edges:
        g.edges[edge]["weight"] = 1

    for node in g.nodes:
        g.nodes[node]["delay"] = 10

    wrapper = GraphWrapper(g)

    found_retiming = False

    while not found_retiming:
        try:
            # calculate a random legal retiming
            r = {}
            # for each node
            for node in wrapper.g.nodes:
                min_in_w_r = np.inf
                min_out_w_r = np.inf
                # for each incoming edge
                for e in wrapper.g.in_edges(node):
                    tail_node = e[0]
                    # calculate the temporary w of the retimed edge
                    # for the retiming of the tail node: if it is not set, take 0
                    w_r = wrapper.g.edges[e]["weight"] - r.get(tail_node, 0)
                    assert w_r >= 0, "the retiming is not legal."

                    # take note of what is the minimum retimed incoming weight
                    if w_r < min_in_w_r:
                        min_in_w_r = w_r

                # for each outcoming edge
                for e in wrapper.g.out_edges(node):
                    head_node = e[1]
                    # calculate the temporary w of the retimed edge
                    # for the retiming of the tail node: if it is not set, take 0
                    w_r = wrapper.g.edges[e]["weight"] + r.get(head_node, 0)
                    assert w_r >= 0, "the retiming is not legal."

                    # take note of what is the minimum retimed incoming weight
                    if w_r < min_out_w_r:
                        min_out_w_r = w_r

                # now we have lowerbound and upperbound for the random choice
                r[int(node)] = randint(-min_in_w_r, min_out_w_r)

            wrapper.set_retimed_graph(r)
            found_retiming = True
        except ValueError:
            print("whoops.")
            found_retiming = False

    print("legal retiming found.")

    _, cp = cp_delta_clock(wrapper.g)
    wrapper.init_WD()

    print(f'previous clock period: {cp}')
    print(f'unique val of D: {len(np.unique(wrapper.D))}')
    print(f'num of nodes: {len(g.nodes)}')
    print(f'num of edges: {len(g.edges)}')

    d_elems_sorted = np.unique(wrapper.D)

    # t0 = time.time()
    # print(wrapper.opt1_initialized(d_elems_sorted))
    t1 = time.time()
    print(wrapper.opt2_initialized(d_elems_sorted))
    t2 = time.time()

    # print(f"opt1 initialized:{t1 - t0}")
    print(f"opt2 initialized:{t2 - t1}")
    return wrapper




if __name__ == '__main__':
    wrapper = test_initialized()
