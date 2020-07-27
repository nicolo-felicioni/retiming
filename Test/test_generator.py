from tqdm import tqdm
import networkx as nx
from random import randint
from Wrappers.GraphWrapper import GraphWrapper
from utils import random_retime

def parse_float(p):
    return str(p).replace('.', 'd')


def generate_test():
    """
    Generate test graphs as dot files.

    It is composed in three phases:

    1. It first generates a backbone_graph, i.e. a cycle (N nodes and N edges). Then, it creates a
    `binomial graph <https://networkx.github.io/documentation/stable/reference/generated/networkx.generators.random_graphs.binomial_graph.html#networkx.generators.random_graphs.binomial_graph>`_
    with a probability p, and merges the two. It is important that the graph has the cyclic backbone because it is required that every node is reachable from any other (for W and D).

    2. After the merge, given two upperbounds upW and upD, respectively for the edge weights and the node delays,
    it initialize every node with a random delay from 1 to upD, and every edge with a random weight from 1 to upW.
    It is **crucial** that the weights of the graph are always >=1, because in this way it is known that the optimal CP
    is the max{d(v)}.

    3. Now, the graph is complete, but it is in its optimal form. To randomize the graph the random_retime function (see utils)
    is used. After that, the graph is saved as dot file.

    These steps are repeated for different ranges of N, p, up_w, up_d. In particular:

    N in [5, 10, 20, 50, 75, 100, 125, 150, 175, 200, 500]

    p in [.0, .05, .1, .2, .3, .5, .75, 1]

    up_w in [1, 5, 10, 100, 1000, 10000]

    up_d in [1, 5, 10, 100, 1000, 10000]

    :return: None
    """

    for N in tqdm([5, 10, 20, 50, 100, 200, 500]):
    # for N in tqdm([75, 125, 150, 175]):
        for p in tqdm([.0, .05, .1, .2, .3, .5, .75, 1]):
            for up_w in [1, 5, 10, 100, 1000, 10000]:
                for up_d in [1, 5, 10, 100, 1000, 10000]:
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
                    nx.set_node_attributes(wrapper.g, wrapper.delay, 'delay')

                    nx.nx_pydot.write_dot(wrapper.g, f"../graph_files/N_{N}_p_{parse_float(p)}_upw_{up_w}_upd_{up_d}")


def create_graph_2100_edge():
    p_dict = {300: 0.0203, 400: 0.01085}

    for N in tqdm(p_dict.keys()):
    # for N in tqdm([75, 125, 150, 175]):
        p = p_dict[N]
        for up_w in [1, 5, 10, 100, 1000, 10000]:
            for up_d in [1, 5, 10, 100, 1000, 10000]:
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
                nx.set_node_attributes(wrapper.g, wrapper.delay, 'delay')

                nx.nx_pydot.write_dot(wrapper.g, f"../graph_files/2100edges/N_{N}_p_{parse_float(p)}_upw_{up_w}_upd_{up_d}")


def create_graph_v_e():
    p_dict = {300: 0.0, 400: 0.0}

    for N in tqdm(p_dict.keys()):
    # for N in tqdm([75, 125, 150, 175]):
        p = 0.0
        for up_w in [1, 5, 10, 100, 1000, 10000]:
            for up_d in [1, 5, 10, 100, 1000, 10000]:
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
                nx.set_node_attributes(wrapper.g, wrapper.delay, 'delay')

                nx.nx_pydot.write_dot(wrapper.g, f"../graph_files/ve/N_{N}_p_{parse_float(p)}_upw_{up_w}_upd_{up_d}")


if __name__ == '__main__':
    create_graph_v_e()
