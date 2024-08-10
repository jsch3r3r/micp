from .utils import _from_template


def generate_scenarios_small(return_info=False):
    """
    Small scenario for testing all algorithms incl. MILP, with more connectivity edges than 'test'
    """
    if return_info:
        return {'group_by': ['r']}  # TODO this is a helper, information could be deduced from the returned values?

    ns = [10]
    r_coms = [0.3]
    n_sls = [5]

    rs = {10: [1, 2, 3]}

    scenarios, ex_config = _from_template(ns, r_coms, n_sls, rs, p_com_keep=0.8)
    ex_config.emicp = {
        'alg_params': {
            'r': {'n': rs},  # number of robots r indexed by number of vertices n
            'fselect_edge': 'select_edge_min_gt'
        }
    }
    ex_config.emicp_milp = {
        'alg_params': {
            'r': {'n': rs}  # number of robots r indexed by number of vertices n
        }
    }
    ex_config.micp_milp = {
        'alg_params': {
            'r': {'n': rs}  # number of robots r indexed by number of vertices n
        }
    }
    ex_config.mmccp = {
        'alg_params': {
            'r': {'n': rs}  # number of robots r indexed by number of vertices n
        }
    }
    return scenarios, ex_config
