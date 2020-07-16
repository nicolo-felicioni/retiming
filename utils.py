import networkx as nx


# checks if the graph is legal after retiming,
# i.e. if all the w(e)>=0
def check_legal_retimed_graph(graph: nx.DiGraph) -> bool:
    for edge in graph.edges:
        if graph.edges[edge]["weight"] < 0:
            return False

    return True


# auxiliary function for CP algorithm.
# it returns only the delta function for each vertex (in the form of a dictionary)
def cp_delta(graph) -> dict:
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

        # for every incoming edge
        for (u, _) in g_zero.in_edges(v):
            if delta[u] > max_delta_u:
                max_delta_u = delta[u]

        delta[v] = graph.nodes[v]["delay"] + max_delta_u

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
