from Wrappers.GraphWrapper import *
import time

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

    t_init = time.time()

    wrapper = GraphWrapper(create_graph_from_d_elist(d, elist))
    wrapper.init_WD()
    print(f"t init:{time.time()-t_init}")
    # init_global_vars(d=d, elist=elist)

    t0 = time.time()
    print(wrapper.opt1())
    t1 = time.time()
    print(wrapper.opt2())
    t2 = time.time()

    print(f"opt1:{t1-t0}")
    print(f"opt2:{t2-t1}")

