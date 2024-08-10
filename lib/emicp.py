import numpy as np
import networkx as nx

from .mmccp import mmccp
from .utils import calc_tour_len, remap_filtered_vertices


def _compute_min_conn_tourgraph(gm, gc, tours):
    """
    Computes the tour graph where the edge weight represents the increase of the maximum tour length (objective z)
    when the two corresponding tours are connected by the vertices v1, v2 in gc that cause the least increase in terms
    of maximum tour length (min_{v1, v2}{max{^c1(v1), ^c2(v2)}}) when adding these detours to v1 in the first tour and
    to v2 in the second tour.
    :param gm:
    :param gc:
    :param tours: assume tours[-1] == [BS], last tour must be BS tour
    :return:
    """
    def _is_bs_tour(_i, _tours):
        return _i == (len(_tours) - 1)

    def _get_bs_tour(_tours):
        return _tours[-1]

    n_t = len(tours)
    n_v = len(gm)

    assert len(tours[-1]) == 1, "Last tour must be the BS tour"

    gt = np.ones((n_t, n_t), dtype=float) * np.inf  # increase in maximum tour length
    gtv = np.ones((n_t, n_t), dtype=int) * -1  # connect-vertices
    gtl = np.zeros((n_t, n_t), dtype=float)  # new length if connect-vertices are inserted
    gti = np.zeros((n_t, n_t), dtype=int)  # indices after which connect-vertex on tour should be inserted

    tlens = [calc_tour_len(t, gm) for t in tours]

    for it1 in range(n_t):
        for it2 in range(it1):
            t1 = tours[it1]
            t2 = tours[it2]
            # pc1 - pc2: potential connectivity pair
            minl = np.inf
            suml = np.inf
            range_pc1 = range(n_v) if not _is_bs_tour(it1, tours) else _get_bs_tour(tours)
            range_pc2 = range(n_v) if not _is_bs_tour(it2, tours) else _get_bs_tour(tours)

            for pc1 in range_pc1:
                for pc2 in range_pc2:
                    if gc[pc1, pc2] or (pc1 == pc2):  # connectivity edge, or the same vertex
                        for iv1_1 in range(len(t1)):
                            iv1_2 = (iv1_1 + 1) % len(t1)
                            for iv2_1 in range(len(t2)):
                                iv2_2 = (iv2_1 + 1) % len(t2)

                                # Is the condition required? Just the BS vertex is considered when BS tour.
                                old1 = gm[t1[iv1_1], t1[iv1_2]] if not _is_bs_tour(it1, tours) else 0
                                old2 = gm[t2[iv2_1], t2[iv2_2]] if not _is_bs_tour(it2, tours) else 0

                                new1 = gm[t1[iv1_1], pc1] + gm[pc1, t1[iv1_2]] if not _is_bs_tour(it1, tours) else 0
                                new2 = gm[t2[iv2_1], pc2] + gm[pc2, t2[iv2_2]] if not _is_bs_tour(it2, tours) else 0

                                d1 = new1 - old1
                                d2 = new2 - old2

                                tle1 = tlens[it1] + d1
                                tle2 = tlens[it2] + d2

                                if (max(tle1, tle2) < minl) or \
                                   ((max(tle1, tle2) == minl) and (tle1 + tle2 < suml)):
                                    minl = max(tle1, tle2)
                                    suml = tle1 + tle2
                                    gt[it1, it2] = d1
                                    gt[it2, it1] = d2
                                    gtv[it1, it2] = pc1  # (pc1, pc2)
                                    gtv[it2, it1] = pc2  # (pc2, pc1)
                                    gtl[it1, it2] = tle1
                                    gtl[it2, it1] = tle2
                                    gti[it1, it2] = iv1_1
                                    gti[it2, it1] = iv2_1

    return gt, gtv, gtl, gti


def select_edge_min_gt(gt: np.ndarray, gtl: np.ndarray, tree: nx.Graph):
    # only consider edges not already in the tree and
    #  edges that do not form a cycle (TODO maybe there is a more efficient code)
    gt_ = gt.copy()
    tree_ = tree.copy()
    gt_[np.eye(gt_.shape[0], dtype=bool)] = np.inf
    for u in range(len(gt)):
        for v in range(u):
            if tree_.has_edge(u, v):
                gt_[u, v] = np.inf
                gt_[v, u] = np.inf
            else:
                tree_.add_edge(u, v)
                has_cycle = not nx.is_tree(tree_)
                if has_cycle:
                    gt_[u, v] = np.inf
                    gt_[v, u] = np.inf
                tree_.remove_edge(u, v)

    # find min max extended max length
    m = gt_.min()
    inds = np.where(gt_ == m)  # ([x0, x1, x2, ...], [y0, y1, y2, ...])

    # among min, find min max{l_u, l_v}
    min_l = np.inf
    min_i = (None, None)
    for u, v in zip(*inds):
        new_l = max(gtl[u, v], gtl[v, u])
        if new_l < min_l:
            min_l = new_l
            min_i = (u, v)

    return min_i


reg_select_edge = {
    'select_edge_min_gt': select_edge_min_gt
}


def emicp(gm: np.ndarray, gc: np.ndarray, vs: np.ndarray, b: int, r: int,
          fselect_edge) -> (list, float, dict):
    """
    Approximation for MICP (minimum idleness connectivity-constrained patrolling).
    Algorithm:
    1. Compute cycle cover in gm[vs] (only sensing locations) with mmccp() [1]
    2. Extend cycle cover such that the tours build a connected tree

    [1] Yu and Liu, "Improved approximation algorithms for some min-max and minimum cycle cover problems",
    Theoretical Computer Science 654, 2016
    :param gm: Undirected complete graph with travel distances on the edges (inf: no path).
    :param gc: Undirected 0/1 graph with edges indicating connectivity (0: disconnected, 1: connected).
    :param vs: List of sensing location indices in gm/gc
    :param b: Base station index in gm/gc  # TODO
    :param r: Number of robots (cycles)
    :param fselect_edge:
    :return:
    """

    def _insert_vertex(tour, vertex, index):
        if vertex not in tour:
            tour.insert(index + 1, vertex)  # if __index >= len(tour), the item is appended
        return tour

    gm_vs = gm[:, vs][vs, :]  # filter the columns then filter the rows
    tours, tlens = mmccp(gm_vs, r)
    tours = remap_filtered_vertices(tours, vs)
    r = len(tours)  # len(tours) can be smaller than r (TODO check this, postprocessing by splitting tours should always decrease max tour length)
    tours.append([b])  # insert BS tour
    tlens.append(0)

    tree = nx.Graph()
    for _ in range(r):
        gt, gtv, gtl, gti = _compute_min_conn_tourgraph(gm, gc, tours)
        ru, rv = fselect_edge(gt, gtl, tree)
        tours[ru] = _insert_vertex(tours[ru], gtv[ru, rv], gti[ru, rv])
        tours[rv] = _insert_vertex(tours[rv], gtv[rv, ru], gti[rv, ru])
        tree.add_edge(ru, rv)

    assert nx.is_tree(tree), "'tree' is not a tree"
    assert len(tree.edges) == r, "len(tree.edges) == {}, r == {}".format(len(tree.edges), r)

    tlens_z = [calc_tour_len(t, gm) for t in tours]
    assert tlens_z[-1] == 0
    assert np.all(np.array(tlens_z) >= np.array(tlens))
    z = max(tlens_z)

    # return the expected standard format for the algorithms:
    sol_dict = dict()
    sol_dict['z'] = z
    sol_dict['tours'] = tours[:-1]  # remove BS tour
    return tours[:-1], z, sol_dict
