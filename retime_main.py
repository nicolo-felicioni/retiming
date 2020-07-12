
import numpy as np
import networkx as nx


# initialization global vars
g = None
W, D = None, None


def WD() -> (np.array, np.array):
    # makes a copy of the original graph
    # could it be expensive?
    new_g = g.copy()
    n_vertices = new_g.number_of_nodes()
    W = np.empty((n_vertices, n_vertices), dtype=np.int)
    D = np.empty((n_vertices, n_vertices))

    # 1. Weight each edge u-e->_ in E with the ordered
    # pair (w(e), -d(u))

    for e in g.edges:
        # weight of edge e
        w_e = g.edges[e]["weight"]
        # delay of node u (i.e. e[0] since e is a tuple)
        d_u = g.nodes[e[0]]["delay"]

        # weight in the new graph is (w(e), -d(u))
        new_g.edges()[e]['weight'] = (w_e, -d_u)

    # 2. compute the all-pairs shortest paths
    # add of the weights: componentwise addition
    # comparison of the weights: lexicographic order

    path_len = dict(nx.all_pairs_dijkstra_path_length(new_g))

    # 3. For each u,v vertices, their shortest path weight is (x,y)
    # set W(u,v) = x, D(u,v) = d(v) - y

    for u in new_g.nodes:
        for v in new_g.nodes:
            (x, y) = path_len[u][v]
            W[u, v] = x
            D[u, v] = new_g.nodes[v]["delay"] - y

    return W, D


# Bellman-Ford test for feasibility
# constraints:
# 1. r(u) - r(v) <= w(e), for all e in E s.t. (u)-e->(v)
# 2. r(u) - r(v) <= W(u,v) - 1, for all u,v in V s.t. D(u,v) > c

def test_feasibility_BF(c) -> (bool, dict):

    is_feasible = True
    r = None

    # 1.
    # reverse returns a copy of a graph with edges reversed
    constraint_graph = g.reverse()

    # 2.
    # find all u, v in V s.t. D(u,v) > c
    pairs_reversed_arr = np.argwhere(D > c)
    # for each u, v add an edge e to the graph s.t. (v)-e->(u)
    # with weight w(e) = W(u,v) - 1
    for u_v in pairs_reversed_arr:
        u = u_v[0]
        v = u_v[1]
        if (v, u) not in constraint_graph.edges:
            constraint_graph.add_edge(v, u)
        constraint_graph.edges[v, u]["weight"] = W[u, v] - 1

    # add a 'dummy' node linked to every other node
    # weights of the edges = 0

    for node in list(constraint_graph.nodes):
        constraint_graph.add_edge("dummy", node)
        constraint_graph.edges["dummy", node]["weight"] = 0

    # try to solve the LP problem
    # if not solvable, throws NetworkXUnbounded exception
    try:
        # is solvable
        r, _ = nx.algorithms.single_source_bellman_ford(constraint_graph, "dummy")
        is_feasible = True
    except nx.exception.NetworkXUnbounded:
        # not solvable
        is_feasible = False

    return is_feasible, r


def test_feasibility_BF_new(c) -> (bool, dict):

    is_feasible = True
    r = None

    # 1.
    # reverse returns a copy of a graph with edges reversed
    constraint_graph = g.reverse()

    # 2.
    # find all u, v in V s.t. D(u,v) > c
    pairs_reversed_arr = np.argwhere(D > c)
    # for each u, v add an edge e to the graph s.t. (v)-e->(u)
    # with weight w(e) = W(u,v) - 1
    for u_v in pairs_reversed_arr:
        u = u_v[0]
        v = u_v[1]
        if (v, u) not in constraint_graph.edges:
            constraint_graph.add_edge(v, u)
        constraint_graph.edges[v, u]["weight"] = W[u, v] - 1

    # add a 'dummy' node linked to every other node
    # weights of the edges = 0
    new_node = max(constraint_graph.nodes) + 1
    for node in list(constraint_graph.nodes):
        constraint_graph.add_edge(new_node, node)
        constraint_graph.edges[new_node, node]["weight"] = 0

    # try to solve the LP problem
    # if not solvable, throws NetworkXUnbounded exception
    try:
        # is solvable
        r, _ = nx.algorithms.single_source_bellman_ford(constraint_graph, new_node)
        is_feasible = True
    except nx.exception.NetworkXUnbounded:
        # not solvable
        is_feasible = False

    return is_feasible, r


def binary_search_minimum(d_elems_sorted: np.array):

    minimum = np.inf
    saved_r = None

    low = 0
    high = len(d_elems_sorted) - 1
    mid = 0

    while low <= high:

        mid = (high + low) // 2

        # Check if x is present at mid
        is_feasible, r = test_feasibility_BF(d_elems_sorted[mid])
        if is_feasible:
            # if d_elems_sorted[mid] < minimum:
            #     minimum = d_elems_sorted[mid]
            minimum = d_elems_sorted[mid]
            saved_r = r
            high = mid - 1
        else:
            low = mid + 1

    # returns the clock period, retiming
    del saved_r["dummy"]
    return minimum, saved_r


def opt1():
    # 1. compute W, D with the WD algorithm -> done

    # 2. sort the elements in the range of D
    # the unique function also sorts the elements
    d_elems_sorted = np.unique(D)

    return binary_search_minimum(d_elems_sorted)


def cp(graph):
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
    delta = {}
    for v in nx.topological_sort(g_zero):
        max_delta_u = 0

        for (u, _) in g_zero.in_edges(v):
            if delta[u] > max_delta_u:
                max_delta_u = delta[u]

        delta[v] = graph.nodes[v]["delay"] + max_delta_u

    # the maximum delta_v for v in V is the clock period
    return delta


def retime(r: dict):
    g_r = g.copy()
    for (u, v) in g.edges:
        g_r.edges[u,v]["weight"] = g.edges[u,v]["weight"] + r.get(v, 0) - r.get(u, 0)

    return g_r


def feas(c) -> (bool, dict):
    # for each vertex v in V, set r(v)=0
    r = {}
    # repeat |V|-1 times:
    for _ in range(len(g.nodes) - 1):
        g_r = retime(r)
        delta = cp(g_r)
        for v in delta.keys():
            if delta[v] > c:
                r[v] = r.get(v, 0) + 1

    g_r = retime(r)
    delta = cp(g_r)

    is_feasible = ((max(delta.values())) <= c)

    return is_feasible, r


def binary_search_minimum_feas(d_elems_sorted):
    minimum = np.inf
    saved_r = None

    low = 0
    high = len(d_elems_sorted) - 1
    mid = 0

    while low <= high:

        mid = (high + low) // 2

        # Check if x is present at mid
        is_feasible, r = feas(d_elems_sorted[mid])
        if is_feasible:
            # if d_elems_sorted[mid] < minimum:
            #     minimum = d_elems_sorted[mid]
            minimum = d_elems_sorted[mid]
            saved_r = r
            high = mid - 1
        else:
            low = mid + 1

    # returns the clock period, retiming
    return minimum, saved_r


def opt2():
    # 1. compute W, D with the WD algorithm -> done

    # 2. sort the elements in the range of D
    # the unique function also sorts the elements
    d_elems_sorted = np.unique(D)

    return binary_search_minimum_feas(d_elems_sorted)



if __name__ == '__main__':
    g = nx.DiGraph()

    d = {
        0: 0,
        1: 3,
        2: 3,
        3: 3,
        4: 3,
        5: 7,
        6: 7,
        7: 7,
    }

    elist = [(0, 1, 1), (1, 7, 0), (1, 2, 1), (2, 6, 0), (2, 3, 1),
             (3, 4, 1), (3, 5, 0), (4, 5, 0), (5, 6, 0), (6, 7, 0), (7, 0, 0)]

    g.add_weighted_edges_from(elist)

    # set the node delays
    nx.set_node_attributes(g, d, "delay")

    W, D = WD()

    print(opt1())
    print(opt2())
