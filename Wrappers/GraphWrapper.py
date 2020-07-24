import time

import numpy as np
import networkx as nx
from utils import *
import utils

class CustomWeight:
    """
    Custom weight class, needed for calculating W and D.

    It has two weights (x,y) and overloads the operator add with componentwise addition, and the
    operator less than with the lexicographic order.
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y

    # adding two objects
    def __add__(self, other_weight):
        if isinstance(other_weight, int):
            print(f"other weight: {other_weight}")
            return self
        else:
            return CustomWeight(self.x + other_weight.x,
                                self.y + other_weight.y)

    # def __eq__(self, other):
    #     # Note: generally, floats should not be compared directly
    #     # due to floating-point precision
    #     return (self.x == other.x) and (self.y == other.y)
    #
    # def __ne__(self, other):
    #     return (self.x != other.x) or (self.y != other.y)
    #
    def __lt__(self, other):
        if self.x == other.x:
            return self.y < other.y
        return self.x < other.x
    #
    # def __gt__(self, other):
    #     return not self.__lt__(other)



class GraphWrapper:
    """
    Wrapper class that contains the nx.DiGraph graph. This class was used for OPT1 and OPT2,
    but a better version of OPT2 is contained in NewGraphWrapper.

    It contains -apart from the graph- the WD matrices (that need to be initialized with the appropriate method first).
    """

    def __init__(self, g: nx.DiGraph, verbose=False):
        """
        Constructor for the wrapper.

        :param g: the graph nx.DiGraph to be wrapped.
        :param verbose: boolean for debug purposes.
        """

        self.g = g
        self.delay = nx.get_node_attributes(g, "delay")
        self.W, self.D = None, None
        self.verbose = verbose


    def init_WD(self):
        """
        Initialization of the two matrices W, D. It calls the function WD.

        :return: None
        """
        self.W, self.D = self.WD()

    def WD(self) -> (np.array, np.array):
        """
        Creates and returns W, D matrices as 2D numpy array.
        W and D are \|V\| x \|V\|.

        #. Weight each edge u-e->_ in E with the ordered pair (w(e), -d(u))

        #. compute the all-pairs shortest paths with a custom Floyd Warshall such that add of the weights: componentwise addition, comparison of the weights: lexicographic order

            * (see CustomWeight class and floyd_warshall_predecessor_and_distance_custom in nx.algorithms.shortest_path.dense.py)

        #. For each u,v vertices, their shortest path weight is (x,y). Set W(u,v) = x, D(u,v) = d(v) - y

        :return: W, D
        """

        # makes a copy of the original graph
        new_g = nx.DiGraph()
        n_vertices = self.g.number_of_nodes()
        W = np.empty((n_vertices, n_vertices), dtype=np.int)
        D = np.empty((n_vertices, n_vertices))

        # 1. Weight each edge u-e->_ in E with the ordered
        # pair (w(e), -d(u))
        new_g.add_weighted_edges_from([(u, v, CustomWeight(self.g.edges[u, v]["weight"], -self.delay[u])) for (u, v) in self.g.edges])

        # 2. compute the all-pairs shortest paths
        # add of the weights: componentwise addition
        # comparison of the weights: lexicographic order
        path_len = dict(nx.floyd_warshall(new_g))

        # 3. For each u,v vertices, their shortest path weight is (x,y)
        # set W(u,v) = x, D(u,v) = d(v) - y

        for u in new_g.nodes:
            for v in new_g.nodes:
                # u = int(u)
                # v = int(v)
                # print(f"u : {u}")
                # print(f"v : {v}")
                cw = path_len[u][v]
                W[int(u), int(v)] = cw.x
                D[int(u), int(v)] = self.delay[v] - cw.y

        return W, D


    def WD_old(self) -> (np.array, np.array):
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
            new_g.edges()[e]['weight'] = CustomWeight(w_e, -d_u)

        # 2. compute the all-pairs shortest paths
        # add of the weights: componentwise addition
        # comparison of the weights: lexicographic order

        path_len = dict(nx.floyd_warshall(new_g))

        # 3. For each u,v vertices, their shortest path weight is (x,y)
        # set W(u,v) = x, D(u,v) = d(v) - y

        for u in new_g.nodes:
            for v in new_g.nodes:
                # u = int(u)
                # v = int(v)
                # print(f"u : {u}")
                # print(f"v : {v}")
                cw = path_len[u][v]
                W[int(u), int(v)] = cw.x
                D[int(u), int(v)] = new_g.nodes[v]["delay"] - cw.y

        return W, D


    def test_feasibility_bf(self, c) -> (bool, dict):
        """
        Bellman-Ford test for feasibility
        constraints:

        #. r(u) - r(v) <= w(e), for all e in E s.t. (u)-e->(v)

        #. r(u) - r(v) <= W(u,v) - 1, for all u,v in V s.t. D(u,v) > c

        The algorithm works in this way:

        #. create constraint_graph, a copy of a graph with edges reversed

        #. find all u, v in V s.t. D(u,v) > c

        #. for each u, v add an edge e to the constraint_graph s.t. (v)-e->(u)

        #. add a "dummy" node linked with weight 0 to every other node

        #. try to solve the LP problem with single_source_bellman_ford from "dummy"

            * if not solvable, throws NetworkXUnbounded exception

        :param c: the clock period to test
        :return: is_feasible: a boolean that says if c is feasible or not; r: if is_feasible, r is the retiming to get c
        """

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
        """
        Finds the minimum clock period feasible given a set of possible clock periods sorted.

        Since the elements are sorted, it uses a binary search.

        It uses test_feasibility_bf to check whether a given cp is feasible or not

        :param d_elems_sorted: a list of possible (unique) clock periods sorted
        :return: the minimum feasible clock period and the corresponding retiming
        """

        minimum = np.inf
        saved_r = None

        low = 0
        high = len(d_elems_sorted) - 1
        mid = 0

        while low <= high:

            mid = (high + low) // 2

            # Check if x is present at mid
            if self.verbose:
                print(f"testing {d_elems_sorted[mid]}")
            is_feasible, r = self.test_feasibility_bf(d_elems_sorted[mid])
            if self.verbose:
                print(f"is {d_elems_sorted[mid]} feasible? {is_feasible}")
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
        """
        OPT-1 algorithm to find the best feasible clock period.

        #. compute W, D with the WD algorithm

        #. sort the elements in the range of D (these elements are contained in the np array d_elems_sorted)

        #. binary search in d the minimum feasible clock period with binary_search_minimum_bf

        :return: see binary_search_minimum_bf
        """
        # 1. compute W, D with the WD algorithm
        if self.W is None or self.D is None:
            print("opt1: initializing W,D...")
            self.init_WD()

        # 2. sort the elements in the range of D
        # the unique function also sorts the elements
        t_start_sort = time.time()
        print("opt1: sorting D...")
        d_elems_sorted = np.unique(self.D)
        print(f"sorted D in {time.time()-t_start_sort}")

        # 3. binary search in d the minimum feasible clock period
        # check with BF
        return self.binary_search_minimum_bf(d_elems_sorted)

    # find the clock period of a graph given
    # returns a dictionary: V -> Natural
    # where CP = max_v{delta(v)}

    # retime the global graph g following function r
    def get_retimed_graph(self, r: dict):
        #g_r = self.g.copy()
        elist = []
        # for each (u)-e->(v):
        # wr(e) = w(e) + r(v) - r(u)
        for (u, v) in self.g.edges:
            elist.append((u, v, self.g.edges[u, v]["weight"] + r.get(v, 0) - r.get(u, 0)))

        g_r = nx.DiGraph()
        g_r.add_weighted_edges_from(elist)
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
            g_r = self.get_retimed_graph(r)
            # calculate deltas for each v through CP algorithm
            delta = cp_delta(g_r)

            for v in delta.keys():
                if delta[v] > c:
                    r[v] = r.get(v, 0) + 1

        g_r = self.get_retimed_graph(r)
        delta, cp = cp_delta_clock(g_r)

        is_feasible = (cp <= c)

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
            if self.verbose:
                print(f"testing {d_elems_sorted[mid]}")
            is_feasible, r = self.feas(d_elems_sorted[mid])
            if self.verbose:
                print(f"is {d_elems_sorted[mid]} feasible? {is_feasible}")
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
        # 1. compute W, D with the WD algorithm
        if self.W is None or self.D is None:
            print("opt2: initializing W,D...")
            self.init_WD()

        # 2. sort the elements in the range of D
        # the unique function also sorts the elements
        t_start_sort = time.time()
        print("opt2: sorting D...")
        d_elems_sorted = np.unique(self.D)
        print(f"sorted D in {time.time() - t_start_sort}")

        # 3. binary search in d the minimum feasible clock period
        # check with FEAS
        return self.binary_search_minimum_feas(d_elems_sorted)

    # applies the retiming r on the graph if it's legal
    def set_retimed_graph(self, r: dict):
        g_r = self.get_retimed_graph(r)
        if not check_legal_retimed_graph(g_r):
            raise ValueError("The retiming is not legal.")
        self.g = g_r

    # opt2 but without the operations of init W, D and d_elems_sorted
    # returns the clock period, retiming
    def opt1_initialized(self, d_elems_sorted):

        # 3. binary search in d the minimum feasible clock period
        # check with BF
        return self.binary_search_minimum_bf(d_elems_sorted)

    # opt2 but without the operations of init W, D and d_elems_sorted
    # returns the clock period, retiming
    def opt2_initialized(self, d_elems_sorted) -> (int, dict):

        # 3. binary search in d the minimum feasible clock period
        # check with FEAS
        return self.binary_search_minimum_feas(d_elems_sorted)


    # OPTIMIZATION FOR FEAS


    # like feas, but with an optimized loop that doesn't re-apply the whole retiming
    # it applies only the changed part
    # returns:
    # is_feasible: bool, is true if c is feasible
    # r: dict, the actual retiming if c is feasible
    def feas_optimized(self, c) -> (bool, dict):
        # for each vertex v in V, set r(v)=0
        r_list = []
        g_r = self.g.copy()
        # repeat |V|-1 times:
        for _ in range(len(self.g.nodes) - 1):
            r = {}
            # calculate deltas for each v through CP algorithm
            delta = self.cp_delta(g_r)

            for v in delta.keys():
                if delta[v] > c:
                    r[v] = r.get(v, 0) + 1
            # TODO DEBUG
            # print(len([e for e in r.values() if e != 0]))
            # retime the graph g following function r
            g_r = utils.get_retimed_graph(g_r, r)
            r_list.append(r)

        r_final = merge_r_list(r_list)
        delta, cp = self.cp_delta_clock(g_r)

        is_feasible = (cp <= c)

        return is_feasible, r_final


    def binary_search_minimum_feas_optimized(self, d_elems_sorted):
        minimum = np.inf
        saved_r = None

        low = 0
        high = len(d_elems_sorted) - 1
        mid = 0

        while low <= high:

            mid = (high + low) // 2

            # Check if x is present at mid
            if self.verbose:
                print(f"testing {d_elems_sorted[mid]}")
            is_feasible, r = self.feas_optimized(d_elems_sorted[mid])
            if self.verbose:
                print(f"is {d_elems_sorted[mid]} feasible? {is_feasible}")
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
    def opt2_optimized(self) -> (int, dict):
        # 1. compute W, D with the WD algorithm
        if self.W is None or self.D is None:
            print("opt2: initializing W,D...")
            self.init_WD()

        # 2. sort the elements in the range of D
        # the unique function also sorts the elements
        t_start_sort = time.time()
        print("opt2: sorting D...")
        d_elems_sorted = np.unique(self.D)
        print(f"sorted D in {time.time() - t_start_sort}")

        # 3. binary search in d the minimum feasible clock period
        # check with FEAS
        return self.binary_search_minimum_feas_optimized(d_elems_sorted)

    def opt2_optimized_initialized(self, d_elems_sorted) -> (int, dict):

        # 3. binary search in d the minimum feasible clock period
        # check with FEAS
        return self.binary_search_minimum_feas_optimized(d_elems_sorted)

    def feas_optimized_numpy(self, c) -> (bool, dict):
        # for each vertex v in V, set r(v)=0
        r_list = []
        g_r = self.g.copy()
        # repeat |V|-1 times:
        for _ in range(len(self.g.nodes) - 1):
            r = {}
            # calculate deltas for each v through CP algorithm
            delta = cp_delta_np(g_r)

            for i, val in enumerate(delta):

                if val > c:
                    r[i] = r.get(i, 0) + 1

            # retime the graph g following function r
            g_r = utils.get_retimed_graph(g_r, r)
            r_list.append(r)



        r_final = merge_r_list(r_list)
        delta, cp = cp_delta_clock(g_r)

        is_feasible = (cp <= c)

        return is_feasible, r_final

    def binary_search_minimum_feas_optimized_np(self, d_elems_sorted):
        minimum = np.inf
        saved_r = None

        low = 0
        high = len(d_elems_sorted) - 1
        mid = 0

        while low <= high:

            mid = (high + low) // 2

            # Check if x is present at mid
            if self.verbose:
                print(f"testing {d_elems_sorted[mid]}")
            is_feasible, r = self.feas_optimized_numpy(d_elems_sorted[mid])
            if self.verbose:
                print(f"is {d_elems_sorted[mid]} feasible? {is_feasible}")
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


    # auxiliary function for CP algorithm.
    # it returns only the delta function for each vertex (in the form of a dictionary)

    def cp_delta(self, graph) -> dict:
        from functools import reduce
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
            max_delta_u = reduce((lambda x, y: max(x, y)), [delta.get(u, 0) for (u, _) in g_zero.in_edges(v)], 0)
            # for (u, _) in g_zero.in_edges(v):
            #     if delta[u] > max_delta_u:
            #         max_delta_u = delta[u]
            delta[v] = self.delay[v] + max_delta_u

        # fill delta dict with the delays of the nodes not present in G0
        for u in graph.nodes:
            if u not in g_zero.nodes:
                delta[u] = self.delay[u]

        # returns the delta dictionary
        return delta


    def cp_delta_clock(self, graph) -> (dict, int):
        delta = self.cp_delta(graph=graph)

        # if delta NOT empty, the maximum delta_v for v in V is the clock period
        # otherwise, it is the maximum delay
        return delta, max(delta.values()) if delta else max([self.delay[node] for node in graph.nodes])


    def feas_optimized_cython(self, c) -> (bool, dict):
        # for each vertex v in V, set r(v)=0
        from MyCython.utils import cp_delta, cp_delta_clock
        r_list = []
        g_r = self.g.copy()
        # repeat |V|-1 times:
        for _ in range(len(self.g.nodes) - 1):
            r = {}
            # calculate deltas for each v through CP algorithm
            delta = cp_delta(g_r, self.delay)

            for v in delta.keys():
                if delta[v] > c:
                    r[v] = r.get(v, 0) + 1
            # TODO DEBUG
            # print(len([e for e in r.values() if e != 0]))
            # retime the graph g following function r
            g_r = utils.get_retimed_graph(g_r, r)
            r_list.append(r)

        r_final = merge_r_list(r_list)
        delta, cp = cp_delta_clock(g_r, self.delay)

        is_feasible = (cp <= c)

        return is_feasible, r_final


    def binary_search_minimum_feas_optimized_cython(self, d_elems_sorted):
        minimum = np.inf
        saved_r = None

        low = 0
        high = len(d_elems_sorted) - 1
        mid = 0

        while low <= high:

            mid = (high + low) // 2

            # Check if x is present at mid
            if self.verbose:
                print(f"testing {d_elems_sorted[mid]}")
            is_feasible, r = self.feas_optimized_cython(d_elems_sorted[mid])
            if self.verbose:
                print(f"is {d_elems_sorted[mid]} feasible? {is_feasible}")
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
