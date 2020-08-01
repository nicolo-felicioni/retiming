from Wrappers.GraphWrapper import *
import time
from random import randint
import utils
from Wrappers.NewGraphWrapper import NewGraphWrapper
import argparse
import os
import matplotlib.pyplot as plt


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', help='Path of the .dot file of the input graph.', required=True)
    parser.add_argument('--opt', '-opt', help='Algorithm to use. Can be 1 (OPT1) or 2 (OPT2)', required=True, type=int)
    parser.add_argument('--output', '-o', help='Path for storing the retimed graph,')
    parser.add_argument('--print_WD', '-wd', action='store_true', help='Option to print the matrices W and D.')
    parser.add_argument('--draw_retimed_graph', '-draw', help='Option for drawing the retimed graph.'
                                                              'Without arguments, it draws both the node ids and delays.\n'
                                                              'You can specify the argument "ids" if you want only node ids printed,'
                                                              'or "delays" if you want only component delays printed.',
                        nargs='?', const=True)
    #parser.add_argument('--use_labels', action='store_true', help='Option for visualizing the labels in the drawing.'
    #                                                              ' If not set, checks for use_delay option.')
    #parser.add_argument('--use_delay', action='store_true', help='Option for visualizing the component delay in the drawing.'
    #                                                             ' If not set, node ids are shown.')
    args = parser.parse_args()
    dict_args = {
        'input': args.input,
        'opt': int(args.opt),
        'draw_retimed_graph': args.draw_retimed_graph,
        'output': args.output,
        'print_WD': args.print_WD,
        #'use_labels': args.use_labels,
        #'use_delay': args.use_delay
    }

    return dict_args


def main(input, opt, draw_retimed_graph, output, print_WD,
         #use_labels, use_delay
         ):


    assert opt == 1 or opt == 2, f"hey there's no OPT-{opt}."

    print("-"*60)
    print("-" * 60)
    print(f"START OPTIMIZATION of {input}")
    print("-"*60)
    print("-" * 60)

    test_graph = read_graph_dot(input)

    if opt == 1:
        wrapper = GraphWrapper(test_graph)
    else:
        wrapper = NewGraphWrapper(test_graph)

    print("initializing WD...")
    t_init = time.time()
    wrapper.init_WD()
    t_wd = time.time()-t_init
    print(f'WD init time:{t_wd}')

    if print_WD:
        print("matrix W:")
        print(wrapper.W)
        print("matrix D:")
        print(wrapper.D)

    _, initial_cp = utils.cp_delta_clock(test_graph)
    print(f"initial Clock Period: {initial_cp}")
    max_delay, n_unique_D, n_nodes, n_edges, n_edges_zero = get_stats(wrapper)

    d_elems_sorted = np.unique(wrapper.D)

    if opt == 1:
        t0 = time.time()
        cp1, r = wrapper.binary_search_minimum_bf(d_elems_sorted)
        t1 = time.time()
    else:
        t0 = time.time()
        cp1, r = wrapper.binary_search_minimum_feas(d_elems_sorted)
        t1 = time.time()

    t_opt = t1 - t0

    print(f"OPT{opt}:{t1 - t0}")
    print(f"OPT{opt} found this clock period: {cp1}")
    print(f"total time: {t_wd+t_opt}")

    # if it is needed to save the retimed graph
    if draw_retimed_graph or output:
        # take the original ids
        original_ids = nx.get_node_attributes(wrapper.g, 'original_ids')

        # set the retiming and the delay attribute
        wrapper.set_retimed_graph(r)
        nx.set_node_attributes(wrapper.g, wrapper.delay, 'delay')
        nx.set_node_attributes(wrapper.g, original_ids, 'original_ids')

        for v in wrapper.g.nodes:
            wrapper.g.nodes[v]['label'] = f'{wrapper.g.nodes[v]["original_ids"]},{wrapper.g.nodes[v]["delay"]}'
            for e in wrapper.g.edges:
                wrapper.g.edges[e]['label'] = wrapper.g.edges[e]['weight']

        if draw_retimed_graph:
            if draw_retimed_graph == 'ids':
                nx.draw(wrapper.g, pos=nx.circular_layout(wrapper.g), with_labels=False, font_weight='bold',
                        node_size=1200)

                nx.draw_networkx_labels(wrapper.g, pos=nx.circular_layout(wrapper.g),
                                        labels=nx.get_node_attributes(wrapper.g, 'original_ids'))
                nx.draw_networkx_edge_labels(wrapper.g, pos=nx.circular_layout(wrapper.g),
                                             edge_labels=nx.get_edge_attributes(wrapper.g, 'weight'))

            elif draw_retimed_graph == 'delays':
                nx.draw(wrapper.g, pos=nx.circular_layout(wrapper.g), with_labels=False, node_size=1200)
                nx.draw_networkx_edge_labels(wrapper.g, pos=nx.circular_layout(wrapper.g),
                                             edge_labels=nx.get_edge_attributes(wrapper.g, 'weight'))
                nx.draw_networkx_labels(wrapper.g, pos=nx.circular_layout(wrapper.g),
                                        labels=nx.get_node_attributes(wrapper.g, 'delay'))
            else:

                nx.draw(wrapper.g, pos=nx.circular_layout(wrapper.g), with_labels=False, node_size=1200)
                nx.draw_networkx_edge_labels(wrapper.g, pos=nx.circular_layout(wrapper.g),
                                             edge_labels=nx.get_edge_attributes(wrapper.g, 'label'))
                nx.draw_networkx_labels(wrapper.g, pos=nx.circular_layout(wrapper.g),
                                        labels=nx.get_node_attributes(wrapper.g, 'label'))

            plt.show()

        if output:
            utils.write_graph_dot(wrapper.g, output)


if __name__ == '__main__':

    dict_args = parse_arguments()

    main(**dict_args)
