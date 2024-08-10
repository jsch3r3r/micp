import numpy as np
from types import SimpleNamespace
from typing import Union
import json
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import skgeom as sg


def is_symmetric(m):
    return np.allclose(np.tril(m).transpose(), np.triu(m))


def symmetric_random(rng, shape):
    r = rng.random(shape)
    r = np.triu(r, 1) + np.triu(r).transpose()
    return r


def generate_micp_instance(opts: Union[dict, SimpleNamespace], rng: np.random.RandomState, copy_attribs=False):
    """
    Vertices are generated in 2D space.
    :param copy_attribs:
    :param rng:
    :param opts: n, n_sl, coords_v, mode_com, r_com, p_com, coords_bs, l_area, r
    :return
    """

    def _intersect_with_obs(_v1, _v2):
        if isinstance(_v1, int):
            assert isinstance(_v2, int)
            pv1 = sg.Point2(*xys[_v1])
            pv2 = sg.Point2(*xys[_v2])
        else:
            assert not isinstance(_v2, int)
            pv1 = _v1
            pv2 = _v2
        pv_seg = sg.Segment2(pv1, pv2)
        for obs_segs in sg_obs_segs:
            for obs_seg in obs_segs:
                if sg.intersection(pv_seg, obs_seg) is not None:
                    return True
        return False

    def _in_obs(_v):
        _vx, _vy = xys[_v]
        for _ox, _oy, _ow, _oh in coords_obs:
            if (_vx >= _ox) and (_vy >= _oy) and (_vx <= _ox + _ow) and (_vy <= _oy + _oh):
                return True
        return False

    if isinstance(opts, dict):
        opts = SimpleNamespace(**opts)

    # --- Begin options ---
    n = opts.n  # V_S + V_C + 1  # total vertex number
    # r = opts.r  # number of robots
    try:
        n_sl = opts.n_sl  # number of SLs
    except AttributeError:
        n_sl = n - 1
    if (n_sl is None) or (n_sl <= 0):
        n_sl = n - 1

    # vertex coordinates
    try:
        coords_v = np.array(opts.coords_v)
        assert coords_v.shape == (n - 1, 2)  # without BS
    except AttributeError:
        coords_v = None

    # obstacle coordinates
    try:
        coords_obs = np.array(opts.coords_obs)
    except AttributeError:
        coords_obs = None
    try:
        mode_com = opts.mode_com  # mode of com edge generation
    except AttributeError:
        mode_com = 'range'

    r_com = np.inf
    p_com = 1.
    if mode_com == 'range':
        try:
            r_com = opts.r_com  # com range
        except AttributeError:
            r_com = np.inf
    elif mode_com.startswith('rand_edge'):
        try:
            p_com = opts.p_com  # probability of edge
        except AttributeError:
            p_com = 0.5

    try:
        p_com_keep = opts.p_com_keep
    except AttributeError:
        p_com_keep = 1.

    try:
        coords_bs = opts.coords_bs  # BS coordinates (in true dimension), random if not specified (None, not existing)
        assert len(coords_bs) == 2
    except AttributeError:
        coords_bs = None

    l_area = None
    if coords_v is None:
        try:
            l_area = opts.l_area  # side length of 2D area
        except AttributeError:
            l_area = 1.
    try:
        coord_gen_type = opts.coord_gen_mode
    except AttributeError:
        coord_gen_type = 'uniform'  # clusters

    try:
        coord_gen_opts = opts.coord_gen_opts
    except AttributeError:
        if coord_gen_type == 'clusters':
            coord_gen_opts = {'n_clusters': 1, 'std': l_area/2.}
        else:
            coord_gen_opts = dict()
    # --- End options ---

    # assumes the first vertices are the SL
    vs = np.zeros((n,), dtype=bool)
    vs[range(n_sl)] = True
    b = n - 1  # BS vertex

    gm = np.zeros((n, n), dtype=float)
    # gc = np.zeros((n, n), dtype=bool)

    if coords_v is None:
        xys = rng.random((n, 2)) * l_area
        if coords_obs is not None:
            for i in range(n):
                while _in_obs(i):
                    xys[i] = rng.random((1, 2)) * l_area
    else:
        xys = np.concatenate([coords_v[:b, :], [[0, 0]], coords_v[b:]], axis=0)
        assert len(xys) == n

    if coords_bs is not None:
        xys[b, :] = coords_bs

    if coords_obs is not None:
        assert all([not _in_obs(v) for v in range(n)])

    for i1, xy1 in enumerate(xys):
        for i2 in range(i1):
            xy2 = xys[i2]
            d = np.linalg.norm(xy1 - xy2, 2)
            gm[i1, i2] = d
            gm[i2, i1] = d

    # we compute gc based on the direct straight line distance and obstacles (therefore need gm before path computation)
    if mode_com == 'range':
        gc = gm <= r_com  # also ensures that two robots at the same vertex can communicate, i.e. gc[i, i] == True
    elif mode_com == 'rand_edge':
        gc = symmetric_random(rng, gm.shape) <= p_com
    elif mode_com == 'rand_edge_range':
        m = np.max(gm)
        assert m < np.inf  # TODO if edges have inf weight, the following code has to be adapted?
        p_com_dist = ((m - gm) / m) * p_com  # gm[i, j] == m => p_com_dist[i, j] == 0, gm[i, j] == 0 => p_com_dist[i, j] == p_com
        gc = symmetric_random(rng, gm.shape) <= p_com_dist
    else:
        raise ValueError("mode_com '{}' not supported".format(mode_com))

    if p_com_keep < 1.:
        gc = gc & (symmetric_random(rng, gm.shape) <= p_com_keep)  # shorthand for np.logical_and()
    gc[np.eye(gc.shape[0], dtype=bool)] = True  # ensure two robots at the same vertex can communicate

    if coords_obs is not None:
        sg_obs_points = [(sg.Point2(o[0], o[1]),
                          sg.Point2(o[0]+o[2], o[1]),
                          sg.Point2(o[0]+o[2], o[1]+o[3]),
                          sg.Point2(o[0], o[1]+o[3])) for o in coords_obs]
        sg_obs_segs = [(sg.Segment2(o[0], o[1]),
                        sg.Segment2(o[1], o[2]),
                        sg.Segment2(o[2], o[3]),
                        sg.Segment2(o[3], o[0])) for o in sg_obs_points]
        for v1 in range(len(gc)):
            for v2 in range(v1):
                if _intersect_with_obs(v1, v2):
                    gc[v1, v2] = 0
                    gc[v2, v1] = 0

    assert is_symmetric(gc)

    if coords_obs is not None:
        def _is_same_seg(_ind1, _ind2):
            if not (_ind1 > n, _ind2 > n):
                return False
            if not ((_ind1 - n) // 4) == ((_ind2 - n) // 4):
                return False
            _ind1m, _ind2m = _ind1 % 4, _ind2 % 4
            return ((_ind1m + 1) % 4 == _ind2m) or ((_ind2m + 1) % 4 == _ind1m)

        sg_obs_points = [(sg.Point2(o[0], o[1]),
                          sg.Point2(o[0]+o[2], o[1]),
                          sg.Point2(o[0]+o[2], o[1]+o[3]),
                          sg.Point2(o[0], o[1]+o[3])) for o in coords_obs]
        sg_obs_segs = [(sg.Segment2(o[0], o[1]),
                        sg.Segment2(o[1], o[2]),
                        sg.Segment2(o[2], o[3]),
                        sg.Segment2(o[3], o[0])) for o in sg_obs_points]
        all_points = []  # list of points not required
        g = nx.Graph()
        for xy in xys:
            all_points.append(sg.Point2(*xy))
        for ps in sg_obs_points:
            for p in ps:
                all_points.append(p)
        for i1, p1 in enumerate(all_points):
            for i2 in range(i1):
                p2 = all_points[i2]
                # we can move along an edge of an obstacle
                if _is_same_seg(i1, i2) or (not _intersect_with_obs(p1, p2)):
                    g.add_edge(i1, i2, weight=np.sqrt(sg.squared_distance(p1, p2)))

        gm = nx.floyd_warshall_numpy(g)
        gm = gm[:, :n][:n, :]

    assert is_symmetric(gm)

    if copy_attribs:
        ret = SimpleNamespace(**vars(opts))
    else:
        ret = SimpleNamespace()

    # These attributes define the MICP problem:
    ret.gm = gm
    ret.gc = gc
    ret.vs = vs
    ret.b = b
    if hasattr(opts, 'r'):
        ret.r = opts.r  # in case copy_attribs == False, the result should also contain the number of robots

    # Additional attributes for debug, visualization
    ret.xys = xys

    return ret


def make_json_serializable(d: Union[dict, list, tuple]) -> Union[dict, list, tuple]:
    raise NotImplementedError("Use utils.JSONEncoder()")


class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        # if isinstance(o, (np.int, np.int16, np.int32, np.int64)):
        #     return int(o)
        # elif isinstance(o, (np.float16, np.float32, np.float64)):
        #     return float(o)
        # elif isinstance(o, np.bool_):  # np.bool_ == np.dtype('bool')
        #     return True if o else False  # bool(o) always returns True
        if hasattr(o, 'item') and (np.prod(o.shape) == 1):
            return o.item()  # returns Python type
        elif isinstance(o, SimpleNamespace):
            return vars(o)

        try:
            iterable = iter(o)
        except TypeError:
            pass
        else:
            return list(iterable)
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, o)


def show_scenario(opts, sol=None, title=None, ax=None):
    if isinstance(opts, dict):
        opts = SimpleNamespace(**opts)

    if not isinstance(opts.vs, np.ndarray):
        opts.vs = np.array(opts.vs)
    if not isinstance(opts.gc, np.ndarray):
        opts.gc = np.array(opts.gc)
    if not isinstance(opts.xys, np.ndarray):
        opts.xys = np.array(opts.xys)
    if hasattr(opts, 'coords_obs') and not isinstance(opts.coords_obs, np.ndarray):
        opts.coords_obs = np.array(opts.coords_obs)

    ext_ax = ax is not None

    if not ext_ax:
        fig, ax = plt.subplots(1, 1)
    ax.plot(opts.xys[opts.vs, 0], opts.xys[opts.vs, 1], 'ob')  # SL
    ax.plot(opts.xys[~opts.vs, 0], opts.xys[~opts.vs, 1], 'ob', fillstyle='none')  # ~SL
    ax.plot(opts.xys[opts.b, 0], opts.xys[opts.b, 1], 'sb', markersize=8)  # BS
    if title is not None:
        ax.set_title(title)
    for i, (x, y) in enumerate(opts.xys[:]):
        ax.text(x + 0.01, y - 0.01, i)
    for u in range(len(opts.gc)):
        for v in range(u):
            if opts.gc[u, v]:
                ax.plot([opts.xys[u, 0], opts.xys[v, 0]],
                        [opts.xys[u, 1], opts.xys[v, 1]], '--r', linewidth=0.5)
    if hasattr(opts, 'coords_obs'):
        for ox, oy, ow, oh in opts.coords_obs:
            o = patches.Rectangle((ox, oy), ow, oh, color='k', fill=True)
            ax.add_patch(o)
    if sol is not None:
        tours = sol['tours']
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for it, t in enumerate(tours):
            for i1 in range(len(t)):
                i2 = (i1 + 1) % len(t)
                u, v = t[i1], t[i2]
                ax.plot([opts.xys[u, 0], opts.xys[v, 0]],
                        [opts.xys[u, 1], opts.xys[v, 1]], '-', color=colors[it % len(colors)], linewidth=1.5)

    if not ext_ax:
        plt.show()


def check_solution(opts, sol, do_raise=True, ignore_conn=False):
    if isinstance(opts, dict):
        opts = SimpleNamespace(**opts)

    gm = np.array(opts.gm)
    z = sol['z']

    tours = sol['tours']
    # sls = np.array(range(opts.n))[opts.vs]
    smsgs = []

    tlens = [calc_tour_len(t, gm) for t in tours]
    act_r = len([1 for t in tours if len(t) > 0])
    if hasattr(opts, 'r'):
        # assumes that artificial BS tour is not in the solution
        if act_r > opts.r:
            smsg = "More tours ({}) than robots ({})".format(len(tours), opts.r)
            smsgs.append(smsg)
            if do_raise:
                raise ValueError(smsg)

    if not np.isclose(max(tlens), z):
        smsg = "Objective value mismatch: z={}, max(tlens)={}".format(z, max(tlens))
        smsgs.append(smsg)
        if do_raise:
            raise ValueError(smsg)

    # Note: a SL vertex can be on multiple tours as connect vertex
    vs = np.copy(opts.vs)
    for t in tours:
        for v in t:
            vs[v] = False
            if (v == opts.b) and (len(t) == 1):
                smsg = "Single BS tour"
                smsgs.append(smsg)
                if do_raise:
                    raise ValueError(smsg)

    if vs.any():
        smsg = "SLs {} not on any tour".format(np.where(vs)[0])
        smsgs.append(smsg)
        if do_raise:
            raise ValueError(smsg)

    if not ignore_conn:
        gc = np.array(opts.gc)
        gt = nx.Graph()
        for it1 in range(len(tours)):
            for v1 in tours[it1]:
                if (v1 == opts.b) or gc[v1, opts.b]:
                    gt.add_edge(it1, act_r + 1)
                for it2 in range(it1):
                    for v2 in tours[it2]:
                        if gc[v1, v2] or (v1 == v2):
                            gt.add_edge(it1, it2)

        if len(gt.nodes) == 0:
            smsg = "Conn: number of vertices is 0"
            smsgs.append(smsg)
            if do_raise:
                raise ValueError(smsg)
        else:
            if len(gt.nodes) < act_r + 1:
                smsg = "Conn: vertices {} < tour vertices {}".format(len(gt.nodes), act_r + 1)
                smsgs.append(smsg)
                if do_raise:
                    raise ValueError(smsg)

            try:
                if not nx.is_connected(gt):
                    smsg = "Tour graph is not connected"
                    smsgs.append(smsg)
                    if do_raise:
                        raise ValueError(smsg)
            # record error but do not interrupt execution, empty graph exceptions should have been filtered out
            except nx.exception.NetworkXException as e:
                smsg = "nx.exception.NetworkXException: {}".format(e)
                smsgs.append(smsg)
                if do_raise:
                    raise ValueError(smsg)

    return smsgs


def calc_tour_len(tour, g):
    if not isinstance(g, np.ndarray):
        g = np.array(g)

    c = 0
    for i in range(len(tour)):
        v1 = tour[i]
        v2 = tour[(i + 1) % len(tour)]
        c += g[v1, v2]
    return c


def remap_filtered_vertices(tours, vs):
    rtours = []
    vsw = np.where(vs)[0]
    for t in tours:
        rt = []
        for v in t:
            rt.append(vsw[v])
        rtours.append(rt)
    return rtours


def dict_str_to_dict(s):
    ss = [ss.strip().split(':') for ss in s.split(',')]
    return {kv[0]: ':'.join(kv[1:]) for kv in ss}
