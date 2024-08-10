import json
import sys
from argparse import ArgumentParser
from typing import Union, Tuple

from lib.utils import generate_micp_instance, JSONEncoder


def _check_str_to_tuple(s: Union[str, Tuple], t):
    if isinstance(s, tuple):
        return s
    if t is None:
        def t(x): return x
    return tuple(t(ss.strip()) for ss in s.split(','))


def generate_env(opts, rng):
    opts.coords_bs = _check_str_to_tuple(opts.coords_bs, float)

    env = generate_micp_instance(opts=opts, rng=rng, copy_attribs=True)

    return env


def generate_envs(scenario_opts, rng, n_per_conf):
    envs = []
    for j, opt in enumerate(scenario_opts):
        i_envs = []
        for i in range(n_per_conf):
            env = generate_env(opt, rng)
            env.__rnd_id__ = i
            env.__scn_id__ = j
            i_envs.append(env)
        envs.append(i_envs)
    return envs


if __name__ == '__main__':
    _parser = ArgumentParser()
    _parser.add_argument('--out_dir', default='scenarios')
    _parser.add_argument('--n', type=int)
    _parser.add_argument('--n_sl', type=int)
    _parser.add_argument('--r', type=int)
    _parser.add_argument('--l_area', type=float, default=1.0)
    _parser.add_argument('--mode_com', type=str)
    _parser.add_argument('--r_com', type=float, default=0.0)
    _parser.add_argument('--p_com', type=float, default=0.5)
    _parser.add_argument('--p_com_keep', type=float, default=1, help='Keep com edge with certain probability')
    _parser.add_argument('--coords_bs', type=str, default='0.0,0.0')
    _parser.add_argument('--seed', type=int, default=122)

    _args = _parser.parse_args(sys.argv[1:])

    _env = generate_env(_args, _args.seed)

    if _args.out_dir:
        with open(_args.out_dir, 'w') as fp:
            json.dump(vars(_env), fp, cls=JSONEncoder)
