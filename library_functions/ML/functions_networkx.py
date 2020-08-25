# normal libraries
import networkx as nx
from matplotlib import pyplot as plt  # plotting


# my libraries


# other files

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def draw_graph(graph, coord, node_size, node_color, cmap):
    plot = nx.draw_networkx_nodes(graph,
                                  coord,
                                  node_size=node_size,
                                  node_color=node_color,
                                  cmap=cmap,
                                  with_labels=False)
    edges = nx.draw_networkx_edges(graph, coord, width=0.65)
    plt.colorbar(plot)
    return


def draw_spring(graph, node_size, node_color, title, cmap='coolwarm'):
    fig = plt.figure(figsize=(12, 5))
    coord = nx.spring_layout(graph, iterations=100, seed=42)
    draw_graph(graph,
               coord,
               node_size=node_size,
               node_color=node_color,
               cmap=cmap)
    plt.title(title + ", spring layout.", y=1.02, fontsize=20)
    return


def draw_spectral(graph, node_size, node_color, title, cmap='coolwarm'):
    fig = plt.figure(figsize=(12, 5))
    coord = nx.spectral_layout(graph)
    draw_graph(graph,
               coord,
               node_size=node_size,
               node_color=node_color,
               cmap=cmap)
    plt.title(title + ", spectral layout.", y=1.02, fontsize=20)
    return


################### only this one is important!
def draw(graph, node_size, node_color, title, spectral=True, spring=True, pos=None, cmap='coolwarm'):
    if spectral:
        draw_spectral(graph, node_size, node_color, title, cmap=cmap)
    if spring:
        draw_spring(graph, node_size, node_color, title, cmap=cmap)
    if pos is not None:
        draw_graph(graph,
                   pos,
                   node_size=node_size,
                   node_color=node_color,
                   cmap=cmap)
    return
