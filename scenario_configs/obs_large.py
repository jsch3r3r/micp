import numpy as np

from .utils import _from_template


def generate_scenario_obs_large(return_info=False):
    """
    Large scenario for testing emicp algorithm, env with obstacles and mulitple clusters of sensing locations.
    A combination for each r and n_clusters will be solved.
    """
    if return_info:
        return {'group_by': ['n', 'r']}  # TODO this is a helper, information could be deduced from the returned values?

    ns = [10,  20,  30,   40,   50,  60,  70,  80,  90,  100]
    r_coms = list(np.linspace(0.2, 0.02, len(ns)))
    n_sls = [int(round(n * 0.4)) for n in ns]

    rs = { 10: [2, 4, 6, 8, 10],
           20: [2, 4, 6, 8, 10],
           30: [2, 4, 6, 8, 10],
           40: [2, 4, 6, 8, 10],
           50: [2, 4, 6, 8, 10],
           60: [2, 4, 6, 8, 10],
           70: [2, 4, 6, 8, 10],
           80: [2, 4, 6, 8, 10],
           90: [2, 4, 6, 8, 10],
          100: [2, 4, 6, 8, 10]}

    scenarios, ex_config = _from_template(ns, r_coms, n_sls, rs, p_com_keep=1)

    def _make_obs_cross(_x, _y):
        # (x, y, w, h):
        return [(_x-0.1, _y-0.01, 0.2, 0.02),
                (_x-0.01, _y-0.1, 0.02, 0.2)]

    # all_opts = []
    for opts in scenarios:
        opts.coords_obs = \
                 _make_obs_cross(0.5, 0.5) + \
                 _make_obs_cross(0.15+0.1, 0.15+0.1) + \
                 _make_obs_cross(0.15+0.2+0.3+0.1, 0.15+0.1) + \
                 _make_obs_cross(0.15+0.2+0.3+0.1, 0.15+0.2+0.3+0.1) + \
                 _make_obs_cross(0.15+0.1, 0.15+0.2+0.3+0.1)

    ex_config.emicp = {
        'alg_params': {
            'r': {'n': rs},  # number of robots r indexed by number of vertices n
            'fselect_edge': 'select_edge_min_gt'
        }
    }
    ex_config.mmccp = {
        'alg_params': {
            'r': {'n': rs}  # number of robots r indexed by number of vertices n
        }
    }
    return scenarios, ex_config
