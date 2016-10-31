import networkx as nx
import matplotlib.pyplot as plt
from random import random, shuffle

w=4;h=4

def create_graph(graph, sink=None, source=None):
    """
    build a networkx graph from a list of edges.
    Edges are 2_uples of vertices. A vertex can be any object.
    :param graph: list of edges
    :param sink, source : nodes
    :return: networkx graph, list of node colors
    """
    #create graph.
    # G is a dictionary with nodes as keys.
    # G[a] is a dictionary with nodes as keys
    # G[a][b] is a dictionary of edge (a,b) data as pairs name:value
    G = nx.Graph()


    # add edges
    for i,edge in enumerate(graph):
        G.add_edge(edge[0], edge[1], weight=i*i)

    # add source and sink
    if sink != None:
        G.add_node(sink)
        for n in G:
            G.add_edge(sink, n, weight=2000*n[0])
    if source != None:
        G.add_node(source)
        for n in G:
            G.add_edge(source, n, weight=2000*n[1])
        G.remove_edge(sink, source)

    # add color and list attributes to nodes
    nx.set_node_attributes(G, 'concat', {n:[] for n in G})
    nx.set_node_attributes(G, 'pixel', (0,0,0))

    return G,source, sink

def color_list(G, partition):
    """
    Given a partition of nodes, colorize classes. All nodes in a class get the same color.
    If there is less than 8 classes, class colors are all different.
    The function returns a list of colors that can be used as node_color parameter in
    draw_networkx.

    :param G: graph
    :param partition: partition of the set of nodes
    :return: list of colors
    """
    tmp_dict = {}
    colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k', 'w']
    for i in range(len(partition)):
        for n in partition[i]:
            tmp_dict[n] = colors[i%8]
    node_color = [tmp_dict[n] for n in G]
    return node_color

def sigmaW(G, v):
    """
    compute the sum of the weights for all edges adjacent to vertex v
    :param G: graph
    :param v: vertex
    :return: sum of weights
    """
    l= G.edges(v,data=True)                     # get list of 3-uples (v,x, {'weight':w})
    s= sum(e[2]['weight'] for e in l)
    return s

def draw_graph(G, node_color):
    """
    draw graph
    :param G: graph
    :param node_color: list of node colors
    :return:
    """
    #pos = nx.shell_layout(G)
    pos = dict((n, (n[0]+0.4*random(), n[1]+0.4*random())) for n in G.nodes())                           # nodes layout : grid
    node_labels = {v: str(v[0])+str(v[1]) for v in G.nodes()}        # node labels : index
    nx.draw_networkx(G, pos, labels=node_labels, node_color=node_color)
    edge_labels = {e[0:2]: '{}'.format(e[2]['weight']) for e in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.show()


def join_nodes(G, a, b, rm=False):
    """
    join two nodes a and b. Raise an error if a and b are not connected
    by a edge e. The common edge e is deleted. All other edges (a,x) connected to a are made
    adjacent to b. If there is already a edge (b,x) the corresponding weights are added.
    By default, node a is not removed from the graph.
    :param G:
    :param a:
    :param b:
    :return:
    """
    #remove common edge
    G.remove_edge(a,b)

    # move edges (a,x) to (b,x)
    # and add weights
    l=G.edges(a)                                        # got list of 2-uples (a,x)
    for e in l:
        w1 = G.get_edge_data(*e)['weight']              # weight of edge (a,e[1])
        G.remove_edge(*e)
        if b in G[e[1]]:                                # G[e[1]] is a dict of node:edge data entries
            w2 = G[e[1]][b]['weight']                   # weight of edge (e[1],b)
        else:
            w2 = 0
        G.add_edge(e[1], b, weight=w1+w2)

    # update concat list for b
    G.node[b]['concat'].append(a)
    G.node[b]['concat'].extend(G.node[a]['concat'])

    if rm:
        G.remove_node(a)

def simplify(G):

    modified = True
    keep_st = True

    while modified:
        modified = False
        i = 0
        # build ist of edges and weights
        if keep_st:
            l = [e for e in G.edges_iter(data=True) if not (e[:2] == (source, sink) or e[:2] == (sink, source))]
        else:
            l = list(G.edges_iter(data=True))

        if len(l) <= 1:
            if keep_st or len(l) == 0:
                break

        shuffle(l)
        e = l[i]  # get 1st edge
        while (e[2]['weight'] <= 0.5 * sigmaW(G, e[0]) and e[2]['weight'] <= 0.5 * sigmaW(G, e[1]) and i < len(l)):
            e = l[i]
            i += 1

        if (e[2]['weight'] > 0.5 * sigmaW(G, e[0]) or e[2]['weight'] > 0.5 * sigmaW(G, e[1])):

            if keep_st:
                if e[0] != sink and e[0] != source:
                    a = e[0]
                    b = e[1]
                elif e[1] != sink and e[1] != source:
                    b = e[0]
                    a = e[1]
            else:
                a = e[0]
                b = e[1]

            print 'collapse', e[0], '-->', e[1]
            join_nodes(G, a, b)
            modified = True

            print 'verif', G.edges(a)

            for b in G.nodes():
                if G[b]:
                    print b, ':', G.node[b]['concat']
                    # draw_graph(G)
        else:
            print "no more reduction found"
        #if source in G[sink]:
            #G[sink][source]['weight']=0

    print 'last****************'


##############
# main code
##############

edges= [((i,j), (i,j+1)) for i in range(w) for j in range (h-1)]+\
       [((i,j), (i+1,j)) for i in range(w-1) for j in range (h)]+\
       [((i,j), (i+1,j+1)) for i in range(w-1) for j in range (h-1)]+\
       [((i,j), (i+1,j-1)) for i in range(w-1) for j in range (1,h)]

# create graph
G, source, sink=create_graph(edges, sink=(2,4), source =(2,-1))

cut_value, partition=nx.minimum_cut(G, source, sink, capacity='weight')
print 'cut', cut_value, partition

node_color=color_list(G,partition) #[ 'y' if (n in partition[0]) else 'g' for n in G ]

draw_graph(G, node_color=node_color)

simplify(G)

cut_value, partition=nx.minimum_cut(G, source, sink, capacity='weight')
print 'cut', cut_value, partition

node_color=color_list(G,partition) #[ 'y' if (n in partition[0]) else 'g' for n in G ]
draw_graph(G, node_color=node_color)

