import numpy as np
import networkx as nx


def cp(graph) -> dict:
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


class Graph:

    def __init__(self, d, elist):
        self.g = nx.DiGraph()

        self.g.add_weighted_edges_from(elist)

        # set the node delays
        nx.set_node_attributes(self.g, d, "delay")

        self.W, self.D = self.WD()


    def WD(self) -> (np.array, np.array):
        # makes a copy of the original graph
        # could it be expensive?
        new_g = self.g.copy()
        n_vertices = new_g.number_of_nodes()
        W = np.empty((n_vertices, n_vertices), dtype=np.int)
        D = np.empty((n_vertices, n_vertices))

        # 1. Weight each edge u-e->_ in E with the ordered
        # pair (w(e), -d(u))

        for e in self.g.edges:
            # weight of edge e
            w_e = self.g.edges[e]["weight"]
            # delay of node u (i.e. e[0] since e is a tuple)
            d_u = self.g.nodes[e[0]]["delay"]

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
    def test_feasibility_bf(self, c) -> (bool, dict):

        is_feasible = True
        r = None

        # 1.
        # reverse returns a copy of a graph with edges reversed
        constraint_graph = self.g.reverse()

        # 2.
        # find all u, v in V s.t. D(u,v) > c
        pairs_reversed_arr = np.argwhere(self.D > c)
        # for each u, v add an edge e to the graph s.t. (v)-e->(u)
        # with weight w(e) = W(u,v) - 1
        for u_v in pairs_reversed_arr:
            u = u_v[0]
            v = u_v[1]
            if (v, u) not in constraint_graph.edges:
                constraint_graph.add_edge(v, u)
            constraint_graph.edges[v, u]["weight"] = self.W[u, v] - 1

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

    def binary_search_minimum_bf(self, d_elems_sorted: np.array):

        minimum = np.inf
        saved_r = None

        low = 0
        high = len(d_elems_sorted) - 1
        mid = 0

        while low <= high:

            mid = (high + low) // 2

            # Check if x is present at mid
            is_feasible, r = self.test_feasibility_bf(d_elems_sorted[mid])
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

    def opt1(self):
        # 1. compute W, D with the WD algorithm -> done

        # 2. sort the elements in the range of D
        # the unique function also sorts the elements
        d_elems_sorted = np.unique(self.D)

        # 3. binary search in d the minimum feasible clock period
        # check with BF
        return self.binary_search_minimum_bf(d_elems_sorted)

    # find the clock period of a graph given
    # returns a dictionary: V -> Natural
    # where CP = max_v{delta(v)}

    # retime the global graph g following function r
    def retime(self, r: dict):
        g_r = self.g.copy()

        # for each (u)-e->(v):
        # wr(e) = w(e) + r(v) - r(u)
        for (u, v) in self.g.edges:
            g_r.edges[u, v]["weight"] = self.g.edges[u, v]["weight"] + r.get(v, 0) - r.get(u, 0)

        return g_r

    # check if c is a feasible clock period
    # returns:
    # is_feasible: bool, is true if c is feasible
    # r: dict, the actual retiming if c is feasible
    def feas(self, c) -> (bool, dict):
        # for each vertex v in V, set r(v)=0
        r = {}
        # repeat |V|-1 times:
        for _ in range(len(self.g.nodes) - 1):
            # retime the graph g following function r
            g_r = self.retime(r)
            # calculate deltas for each v through CP algorithm
            delta = cp(g_r)

            for v in delta.keys():
                if delta[v] > c:
                    r[v] = r.get(v, 0) + 1

        g_r = self.retime(r)
        delta = cp(g_r)

        is_feasible = ((max(delta.values())) <= c)

        return is_feasible, r

    def binary_search_minimum_feas(self, d_elems_sorted):
        minimum = np.inf
        saved_r = None

        low = 0
        high = len(d_elems_sorted) - 1
        mid = 0

        while low <= high:

            mid = (high + low) // 2

            # Check if x is present at mid
            is_feasible, r = self.feas(d_elems_sorted[mid])
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

    # returns the clock period, retiming
    def opt2(self) -> (int, dict):
        # 1. compute W, D with the WD algorithm -> done

        # 2. sort the elements in the range of D
        # the unique function also sorts the elements
        d_elems_sorted = np.unique(self.D)

        # 3. binary search in d the minimum feasible clock period
        # check with FEAS
        return self.binary_search_minimum_feas(d_elems_sorted)

