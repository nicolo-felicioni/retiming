import time

import numpy as np
import networkx as nx
from utils import *
import utils
from functools import reduce


class CustomWeight:
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



class NewGraphWrapper:
    """
    Wrapper class that wraps the nx.DiGraph graph. This class is used for OPT2.

    It wraps -apart from the graph- the WD matrices, component delay (both as dict and as np.arr), weight dict, nodes and edges list, num of nodes.
    """

    def __init__(self, g: nx.DiGraph, verbose=False):
        self.g = g
        self.delay = nx.get_node_attributes(g, "delay")
        self.weight = nx.get_edge_attributes(g, "weight")
        # self.weight_arr = np.array(list(self.weight.items()), dtype=np.int)
        self.nodes = list(g.nodes)
        self.edges = list(g.edges)
        self.num_nodes = len(self.nodes)
        self.delay_arr = np.empty(shape=self.num_nodes)
        for k in self.delay.keys():
            self.delay_arr[k] = self.delay[k]

        self.W, self.D = None, None
        self.verbose = verbose


    def init_WD(self):
        self.W, self.D = self.WD()


    def WD(self) -> (np.array, np.array):
        """
        Creates the matrices W and D.

        :return: W, D matrices as 2D np.array
        """

        # makes a copy of the original graph
        # could it be expensive?
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
                cw = path_len[u][v]
                W[int(u), int(v)] = cw.x
                D[int(u), int(v)] = self.delay[v] - cw.y

        return W, D


    def binary_search_minimum_feas(self, d_elems_sorted):
        """
        Finds the minimum clock period feasible given a set of possible clock periods sorted.

        Since the elements are sorted, it uses a binary search.

        It uses FEAS to check whether a given cp is feasible or not

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


    def feas(self, c):
        """
        Function that checks if the parameter c is feasible as clock period for the graph contained in the wrapper.
        In this version, it works like this:


        #. it creates a matrix r_mat with shape (\|V\|, \|V\|-1) full of zeros, a list edges_zero of edges that have w(e)=0 and a copy of the weight dictionary

        #. for i from 0 to \|V\| - 1:

            * take r = r_mat[:, i], namely the i-th column

            * calculate delta values for every v (see cp_delta)

            * while calculating delta, if delta(v) > c -> r[v]++

            * update weight dictionary with the retiming r and update edges_zero accordingly

        #. calculate cp and delta on the final retiming, looking in weight and if cp <= c, then c is feasible

        :param c: the clock period to be tested
        :return: is_feasible: a boolean that says if c is a feasible clock period; r_final: the final retiming
        """
        r_mat = np.zeros(shape=(self.num_nodes, self.num_nodes-1))
        edges_zero = set([edge for edge in self.edges if self.weight[edge] == 0])
        weight = self.weight.copy()

        for i in range(self.num_nodes-1):
            r = r_mat[:, i]

            # Calculate delta (CP algo) and update r
            g_zero = nx.DiGraph()
            g_zero.add_edges_from(edges_zero)
            delta = {}

            checked_nodes_set = set(g_zero.nodes)
            for v in nx.topological_sort(g_zero):
                # for every incoming edge
                max_delta_u = reduce((lambda x, y: max(x, y)), [delta.get(u, 0) for (u, _) in g_zero.in_edges(v)], 0)
                # for (u, _) in g_zero.in_edges(v):
                #     if delta[u] > max_delta_u:
                #         max_delta_u = delta[u]
                delta[v] = self.delay_arr[v] + max_delta_u
                if delta[v] > c:
                    r[v] = r[v] + 1

            for v in self.nodes:
                if v not in checked_nodes_set:
                    if self.delay_arr[v] > c:
                        r[v] = r[v] + 1

            # retiming and update of edges_zero
            for (u, v) in self.edges:
                weight[u, v] = weight[u, v] + r[v] - r[u]
                if weight[u, v] == 0:
                    edges_zero.add((u, v))
                if (u, v) in edges_zero and weight[u, v] != 0:
                    edges_zero.remove((u, v))

        # Calculate final CP
        g_zero = nx.DiGraph()
        g_zero.add_edges_from(edges_zero)
        delta = {}

        for v in nx.topological_sort(g_zero):

            # for every incoming edge
            max_delta_u = reduce((lambda x, y: max(x, y)), [delta.get(u, 0) for (u, _) in g_zero.in_edges(v)], 0)
            # for (u, _) in g_zero.in_edges(v):
            #     if delta[u] > max_delta_u:
            #         max_delta_u = delta[u]
            delta[v] = self.delay_arr[v] + max_delta_u

        cp = 0
        for v in range(self.num_nodes):
            delta_v = delta.get(v, self.delay_arr[v])
            if delta_v > cp:
                cp = delta_v

        is_feasible = (cp <= c)
        r_final = r_mat.sum(axis=1)

        return is_feasible, r_final
