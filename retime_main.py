from Graph import *


# checks if the graph is legal after retiming,
# i.e. if all the w(e)>=0
def check_legal_retime(graph: nx.DiGraph) -> bool:
    for edge in graph.edges:
        if graph.edges[edge]["weight"] < 0:
            return False

    return True


if __name__ == '__main__':

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

    g = Graph(d, elist)
    # init_global_vars(d=d, elist=elist)

    print(g.opt1())
    print(g.opt2())

