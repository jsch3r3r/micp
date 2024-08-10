import copy
from types import SimpleNamespace


def _from_template(ns, r_coms, n_sls, rs, p_com_keep=0.5):
    ex_config = SimpleNamespace()

    opts_template = SimpleNamespace()
    opts_template.l_area = 1.
    opts_template.mode_com = 'range'
    opts_template.p_com = 1.
    opts_template.p_com_keep = p_com_keep
    opts_template.coords_bs = (0., 0.)

    scenarios = []

    for n, n_sl, r_com in zip(ns, n_sls, r_coms):

        opts = copy.deepcopy(opts_template)
        opts.n = n
        opts.r_com = r_com
        opts.n_sl = n_sl

        scenarios.append(opts)

    return scenarios, ex_config

