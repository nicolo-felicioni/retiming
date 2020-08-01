import networkx as nx
from random import randint
import numpy as np


def check_legal_retimed_graph(graph: nx.DiGraph) -> bool:
    """
    checks if the graph is legal after retiming, i.e. if all the w(e)>=0

    :param graph: graph to check
    :return: boolean, True if retiming is legal, False otherwise
    """
    for edge in graph.edges:
        if graph.edges[edge]["weight"] < 0:
            return False

    return True


# # auxiliary function for CP algorithm.
# # it returns only the delta function for each vertex (in the form of a dictionary)
#
def cp_delta(graph) -> dict:
    # Let G0 be the subgraph of G that contains those edges with w(e)=0
    # elist_zero = []

    #for edge in graph.edges:
    #    if graph.edges[edge]["weight"] == 0:
    #        elist_zero.append(edge)

    # Let G0 be the subgraph of G that contains those edges with w(e)=0
    g_zero = nx.DiGraph()
    g_zero.add_edges_from([edge for edge in graph.edges if graph.edges[edge]["weight"] == 0])

    # By W2, G0 is acyclic. Perform topol. sort s.t. if there is
    # (u)-e->(v) => u < v
    # go through the vertices in that topol. sort order and compute delta_v:
    # a. if there is no iincoming edge to v, delta_v = d(v)
    # b. otherwise,
    # delta_v = d(v) + max_{u in V s.t. (u)-e->(v) incoming and w(e)=0}(delta_u)
    delta = {}
    for v in nx.topological_sort(g_zero):
        max_delta_u = 0

        # for every incoming edge
        for (u, _) in g_zero.in_edges(v):
            if delta[u] > max_delta_u:
                max_delta_u = delta[u]

        delta[v] = graph.nodes[v]["delay"] + max_delta_u

    # fill delta dict with the delays of the nodes not present in G0
    for u in graph.nodes:
        if u not in g_zero.nodes:
            delta[u] = graph.nodes[u]["delay"]

    # returns the delta dictionary
    return delta


def cp_delta_clock(graph) -> (dict, int):
    delta = cp_delta(graph=graph)

    # if delta NOT empty, the maximum delta_v for v in V is the clock period
    # otherwise, it is the maximum delay
    return delta, max(delta.values()) if delta else max([graph.nodes[node]["delay"] for node in graph.nodes])


def create_graph_from_d_elist(d, elist):

    g = nx.DiGraph()
    g.add_weighted_edges_from(elist)
    # set the node delays
    nx.set_node_attributes(g, d, "delay")

    return g


# BUG
# retime the global graph g following function r
def get_retimed_graph_nodes_strategy(graph: nx.DiGraph, r: dict):
    g_r = graph.copy()

    # for each (u)-e->(v):
    #                HEAD   TAIL
    # wr(e) = w(e) + r(v) - r(u)
    # for (u, v) in graph.edges:
    #    g_r.edges[u, v]["weight"] = graph.edges[u, v]["weight"] + r.get(v, 0) - r.get(u, 0)
    for n in r.values():
        for e in g_r.in_edges(n):
            g_r.edges[e[0], e[1]]["weight"] = g_r.edges[e[0], e[1]]["weight"] + r.get(n, 0)

        for e in g_r.out_edges(n):
            g_r.edges[e[0], e[1]]["weight"] = g_r.edges[e[0], e[1]]["weight"] - r.get(n, 0)

    return g_r


# retime the global graph g following function r
def get_retimed_graph(graph: nx.DiGraph, r: dict):

    # for each (u)-e->(v):
    # wr(e) = w(e) + r(v) - r(u)
    # elist = [(u, v, graph.edges[u, v]["weight"] + r.get(v, 0) - r.get(u, 0)) for (u,v) in graph.edges]
    # for (u, v) in graph.edges:
    #     elist.append((u, v, graph.edges[u, v]["weight"] + r.get(v, 0) - r.get(u, 0)))

    g_r = nx.DiGraph()
    g_r.add_weighted_edges_from([(u, v, graph.edges[u, v]["weight"] + r.get(v, 0) - r.get(u, 0))
                                 for (u, v) in graph.edges])
    return g_r


# retime the global graph g following function r
def get_retimed_graph_numpy(graph: nx.DiGraph, r: np.ndarray):

    # for each (u)-e->(v):
    # wr(e) = w(e) + r(v) - r(u)
    # elist = [(u, v, graph.edges[u, v]["weight"] + r.get(v, 0) - r.get(u, 0)) for (u,v) in graph.edges]
    # for (u, v) in graph.edges:
    #     elist.append((u, v, graph.edges[u, v]["weight"] + r.get(v, 0) - r.get(u, 0)))

    g_r = nx.DiGraph()
    g_r.add_weighted_edges_from([(u, v, graph.edges[u, v]["weight"] + r[v] - r[u])
                                 for (u, v) in graph.edges])
    return g_r

# function that merges a list of retimings (dictionaries)
def merge_r_list(r_list):
    r_final = {}
    key_list = []
    for r in r_list:
        key_list += r.keys()

    key_set = set(key_list)
    for k in key_set:
        v = 0
        for r in r_list:
            v = v + r.get(k, 0)

        r_final[k] = v

    return r_final


def random_retime(graph):
    """
    Creates a random retime for a graph that is legal by construction.
    It finds, for each node, the lowerbound and the upperbound for the retiming of that node, and then
    chooses a random integer between lb and ub.
    Here is how the bounds are found:

    #. set r(v):=0 for each v

    #. For each node v:

        * find what is the minimum retimed incoming weight min_in_w_r

        * find what is the minimum retimed outcoming weight min_out_w_r

    #. The retiming is a random integer number between (-min_in_w_r, min_out_w_r)

    :param graph: graph to retime
    :return: a legal retime r
    """

    # calculate a random legal retiming
    r = {}
    # for each node
    for node in graph.nodes:
        min_in_w_r = np.inf
        min_out_w_r = np.inf
        # for each incoming edge
        for e in graph.in_edges(node):
            tail_node = e[0]
            # calculate the temporary w of the retimed edge
            # for the retiming of the tail node: if it is not set, take 0
            w_r = graph.edges[e]["weight"] - r.get(tail_node, 0)
            assert w_r >= 0, "the retiming is not legal."

            # take note of what is the minimum retimed incoming weight
            if w_r < min_in_w_r:
                min_in_w_r = w_r

        # for each outcoming edge
        for e in graph.out_edges(node):
            head_node = e[1]
            # calculate the temporary w of the retimed edge
            # for the retiming of the tail node: if it is not set, take 0
            w_r = graph.edges[e]["weight"] + r.get(head_node, 0)
            assert w_r >= 0, "the retiming is not legal."

            # take note of what is the minimum retimed incoming weight
            if w_r < min_out_w_r:
                min_out_w_r = w_r

        # now we have lowerbound and upperbound for the random choice
        r[int(node)] = randint(-min_in_w_r, min_out_w_r)

    return r


def read_graph_dot_misc_d(path):
    g = nx.DiGraph(nx.nx_pydot.read_dot(path))

    # convert node labels to int
    g = nx.relabel_nodes(g, lambda x: int(x) if x != 'vh' else x)

    if 'vh' in g.nodes:
        list_nodes = list(g.nodes)
        list_nodes.remove('vh')
        max_id = max(list_nodes)
        g = nx.relabel_nodes(g, lambda x: max_id+1 if x == 'vh' else x)

    # convert node delays to float
    for node in g.nodes:
        g.nodes[node]["delay"] = float(g.nodes[node]["d"])

    # convert edge w to int
    for e in g.edges:
        g.edges[e]["weight"] = int(g.edges[e]["w"])

    return g


def read_graph_dot_misc_l(path):

    g = nx.DiGraph(nx.nx_pydot.read_dot(path))

    # convert node labels to int
    g = nx.relabel_nodes(g, lambda x: int(x) if x != 'vh' else x)

    if 'vh' in g.nodes:
        list_nodes = list(g.nodes)
        list_nodes.remove('vh')
        max_id = max(list_nodes)
        g = nx.relabel_nodes(g, lambda x: max_id+1 if x == 'vh' else x)

    # convert node delays to float
    for node in g.nodes:
        g.nodes[node]["delay"] = float(g.nodes[node]["component_delay"])

    # convert edge w to int
    for e in g.edges:
        g.edges[e]["weight"] = int(g.edges[e]["wire_delay"])

    return g


def read_graph_dot(path):
    g = nx.DiGraph(nx.nx_agraph.read_dot(path))

    # convert node labels to int
    mapping = {}
    i = 0
    for n in g.nodes:
        if n not in mapping.keys():
            mapping[n] = i
            i += 1

    g = nx.relabel_nodes(g, mapping)
    inv_mapping = {v: k for k, v in mapping.items()}
    nx.set_node_attributes(g, inv_mapping, 'original_ids')

    # convert node delays to float
    for node in g.nodes:
        g.nodes[node]["delay"] = float(g.nodes[node]["delay"])

    # convert edge w to int
    for e in g.edges:
        g.edges[e]["weight"] = int(g.edges[e]["weight"])

    print(g.name)
    return g

def write_graph_dot(graph, path):
    # inv_mapping = {v: k for k, v in nx.get_node_attributes(graph, "original_ids").items()}
    nx.relabel_nodes(graph, nx.get_node_attributes(graph, "original_ids"), copy=False)
    for (n, d) in graph.nodes(data=True):
        del d["original_ids"]
    nx.nx_pydot.write_dot(graph, path)



def get_stats(wrapper, verbose=True):
    optimal_cp = max([wrapper.g.nodes[node]['delay'] for node in wrapper.g.nodes])
    n_unique_D = len(np.unique(wrapper.D))
    n_nodes = len(wrapper.g.nodes)
    n_edges = len(wrapper.g.edges)
    n_edges_zero = len([e for e in wrapper.g.edges
                        if wrapper.g.edges[e]["weight"] == 0])

    if verbose:
        print(f"Maximum component delay: {optimal_cp}")
        print(f'unique val of D: {n_unique_D}')
        print(f'num of nodes: {n_nodes}')
        print(f'num of edges: {n_edges}')
        print(f'num of edges 0: {n_edges_zero}')

    return optimal_cp, n_unique_D, n_nodes, n_edges, n_edges_zero


def cp_delta_np(graph) -> np.ndarray:
    # Let G0 be the subgraph of G that contains those edges with w(e)=0
    elist_zero = []
    for edge in graph.edges:
        if graph.edges[edge]["weight"] == 0:
            elist_zero.append(edge)

    g_zero = nx.DiGraph()
    g_zero.add_edges_from(elist_zero)

    # By W2, G0 is acyclic. Perform topol. sort s.t. if there is
    # (u)-e->(v) => u < v
    # go through the vertices in that topol. sort order and compute delta_v:
    # a. if there is no iincoming edge to v, delta_v = d(v)
    # b. otherwise,
    # delta_v = d(v) + max_{u in V s.t. (u)-e->(v) incoming and w(e)=0}(delta_u)
    delta = np.empty(shape=len(graph.nodes), dtype=np.int)
    for v in nx.topological_sort(g_zero):
        max_delta_u = 0

        # for every incoming edge
        for (u, _) in g_zero.in_edges(v):
            if delta[u] > max_delta_u:
                max_delta_u = delta[u]

        delta[v] = graph.nodes[v]["delay"] + max_delta_u

    # fill delta dict with the delays of the nodes not present in G0
    for u in graph.nodes:
        if u not in g_zero.nodes:
            delta[u] = graph.nodes[u]["delay"]

    # returns the delta dictionary (np array)
    return delta
