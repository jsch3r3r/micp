import numpy as np
import networkx as nx
from functools import reduce

from .utils import is_symmetric


def christofides(gx: nx.Graph) -> list:
    """

    :param gx: complete graph with path lengths
    :return:
    """
    if len(gx.nodes) == 1:
        return list(gx.nodes)

    mst = nx.minimum_spanning_tree(gx)

    odds = {i for i, v in enumerate(mst.adjacency()) if len(v[1]) % 2 == 1}
    assert len(odds) % 2 == 0, "Number of vertices with odd degree should be even!"
    g_odds = gx.subgraph(odds)
    m = nx.algorithms.matching.min_weight_matching(g_odds)
    for (u, v) in m:
        mst.add_edge(u, v)

    it = nx.eulerian_circuit(mst.to_directed())
    vs = [i[0] for i in it]

    # vis = np.zeros(len(gx.nodes), dtype=bool)
    vis = {v: False for v in gx.nodes}
    tour = []
    for v in vs:
        if not vis[v]:
            tour.append(v)
            vis[v] = True

    return tour


def _split_tour(g: np.ndarray, tour: list, lambd: float) -> (list, list):
    """
    Helper for MMCPP(lambda) which splits a given tour into subtours.
    :param g:
    :param tour:
    :param lambd:
    :return:
    """
    ws = 0.
    tours = [[tour[0]]]
    lens = []
    tourind = 0
    for i in range(1, len(tour)):
        w = g[tour[i - 1], tour[i]]
        if (ws + w) > 5/2. * lambd:
            tours.append([])
            lens.append(ws)
            tourind += 1
            ws = 0
            w = 0

        tours[tourind].append(tour[i])
        ws += w

    if len(lens) < len(tours):
        lens.append(ws)

    for i, t in enumerate(tours):
        w = g[t[-1], t[0]]
        lens[i] += w

    return tours, lens


def _mmccp_split(gx: nx.Graph, lambd: float, k: int) -> (list, list):
    """
    MMCCP(lambda) from [1].

    [1] Yu and Liu, "Improved approximation algorithms for some min-max and minimum cycle cover problems",
    Theoretical Computer Science 654, 2016
    :param gx:
    :param lambd:
    :param k:
    :return:
    """
    gx = gx.copy()

    heavy_es = [e for e, d in gx.edges.items() if d['weight'] > lambd / 2.]
    for e in heavy_es:
        gx.remove_edge(e[0], e[1])
    comps = nx.connected_components(gx)
    tours = []
    lens = []
    for c in comps:
        gcx = gx.subgraph(c)
        tourc = christofides(gcx)
        toursc, lensc = _split_tour(nx.convert_matrix.to_numpy_matrix(gx), tourc, lambd)  # use gx since tourc can contain vertices from the whole graph
        tours.extend(toursc)
        lens.extend(lensc)

    if len(tours) > k:
        raise ValueError("Result has more than {} cycles!".format(k))
    else:
        return tours, lens


def mmccp(g: np.ndarray, k: int) -> (list, list):
    """
    Binary interval search applying MMCCP(lambda) [1].

    [1] Yu and Liu, "Improved approximation algorithms for some min-max and minimum cycle cover problems",
    Theoretical Computer Science 654, 2016
    :param g: complete graph
    :param k: number of tours
    :return: list of subtours
    """
    assert is_symmetric(g), "Adjacency matrix must be symmetric"

    gx = nx.convert_matrix.from_numpy_matrix(g)

    lo, up = 0, g.sum() / 2
    lambd = up
    d = np.inf

    tours, lens = None, None
    while d > 0.1:
        lambd_ = lambd
        try:
            tours, lens = _mmccp_split(gx, lambd, k)
            up = lambd
        except ValueError:
            lo = lambd
        lambd = lo + (up - lo) / 2
        d = np.abs(lambd_ - lambd)

    return tours, lens


def check_mmccp_sol(tours, vs, r):
    # there can be less tours than robots
    # tours contain all SLs
    # tours contain each SL only once
    return len(tours) <= r and \
        ((vs is None) or (set(np.where(vs)[0]) == set(reduce(lambda l1, l2: l1 + l2, tours)))) and \
        len(set(reduce(lambda l1, l2: l1 + l2, tours))) == len(reduce(lambda l1, l2: l1 + l2, tours))
