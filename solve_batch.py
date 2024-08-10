import os
import sys
import argparse
import numpy as np
import json
import copy
from functools import reduce, partial
import multiprocessing as mp
import time
from datetime import datetime

from lib.generate_scenario import generate_envs
from lib.utils import JSONEncoder, check_solution, dict_str_to_dict, remap_filtered_vertices, calc_tour_len
from lib.emicp import emicp, reg_select_edge
from lib.micp_milp import micp_milp, emicp_milp
from lib.mmccp import mmccp


def _prepare_env_list(envs):
    # make flat list of lists
    if isinstance(envs, list) and isinstance(envs[0], list):
        envs = reduce(lambda x, y: x + y, envs)
    return envs


def _mmccp_wrapper(gm: np.ndarray, gc: np.ndarray, vs: np.ndarray, b: int, r: int):
    gm_vs = gm[:, vs][vs, :]  # filter the columns then filter the rows
    tours, _ = mmccp(gm_vs, r)
    tours = remap_filtered_vertices(tours, vs)

    tlens_z = [calc_tour_len(t, gm) for t in tours]
    z = max(tlens_z)

    # return the expected standard format for the algorithms:
    sol_dict = dict()
    sol_dict['z'] = z
    sol_dict['tours'] = tours
    return tours, z, sol_dict


def _prepare_alg_params(alg_name, param_list, env_list, sols_dir, args):
    # def emicp     (gm: np.ndarray, gc: np.ndarray, vs: np.ndarray, b: int, r: int, fselect_edge)
    # def micp_milp (gm: np.ndarray, gc: np.ndarray, vs: np.ndarray, b: int, r: int, save_model='model/model.cip', load_sol=None, save_sol='model/model.sol'):
    # def emicp_milp(gm: np.ndarray, gc: np.ndarray, vs: np.ndarray, b: int, r: int, save_model='model/model.cip', load_sol=None, save_sol='model/model.sol'):
    # -> tuple[list[list[int]], float, dict]
    assert len(param_list) == len(env_list)

    alg_param_list = []
    for i, (param, env) in enumerate(zip(param_list, env_list)):
        d = {
            'gm': np.array(env['gm']),
            'gc': np.array(env['gc']),
            'vs': np.array(env['vs']),
            'b': env['b'],
            'r': param['r']
        }

        if alg_name == 'emicp':
            d['fselect_edge'] = reg_select_edge[param['fselect_edge']]
        elif alg_name.endswith('_milp'):
            # TODO should this be saved? Is mainly for debugging, the actual solution is collected and saved in main loop
            d['save_model'] = os.path.join(sols_dir, '{:03d}.cip'.format(i)) if sols_dir is not None else None
            d['save_sol'] = os.path.join(sols_dir, '{:03d}.sol'.format(i)) if sols_dir is not None else None
            d['solver_params'] = dict_str_to_dict(args.solver_params) if args.solver_params else None
            # d['load_sol'] = 'scenarios/sols/test/micp_milp/009.sol'  # for debugging
        alg_param_list.append(d)

    return alg_param_list


def _prepare_save_params(p, env):
    p_ = copy.deepcopy(p)
    if 'fselect_edge' in p:
        p_['fselect_edge'] = [k for k, v in reg_select_edge.items() if v is p['fselect_edge']][0]  # 'is' returns true if both point to the same object
    p_['n'] = env['n']
    if '__rnd_id__' in env:
        p_.update({'__rnd_id__': env['__rnd_id__']})
    if '__scn_id__' in env:
        p_.update({'__scn_id__': env['__scn_id__']})
    return p_


def prepare_configs(envs, ex_alg_configs):
    all_configs = []
    all_envs = []
    for env in envs:
        iter_envs = []
        iter_configs = []
        for alg_param, param_values in ex_alg_configs['alg_params'].items():  # r: {}

            if isinstance(param_values, dict):  # depends on another env param
                for env_attrib, env_value in param_values.items():  # n: {10:}
                    if env_attrib in env.keys():
                        param_env_values = env_value.get(str(env[env_attrib]), None)
                        if param_env_values is not None:
                            if isinstance(param_env_values, list):
                                if len(iter_configs) == 0:
                                    iter_envs.append(env)
                                    iter_configs.append(dict())

                                iter_value_configs = []
                                iter_value_envs = []
                                for i, iconf in enumerate(iter_configs):
                                    for param_env_value in param_env_values:
                                        iter_value_config = copy.deepcopy(iconf)
                                        iter_value_config[alg_param] = param_env_value
                                        iter_value_configs.append(iter_value_config)
                                        iter_value_envs.append(env)
                                iter_configs.clear()
                                iter_configs.extend(iter_value_configs)
                                iter_envs.clear()
                                iter_envs.extend(iter_value_envs)
                            else:
                                pass

            elif isinstance(param_values, list):  # iterate over all values
                if len(iter_configs) == 0:
                    iter_envs.append(env)
                    iter_configs.append(dict())

                iter_value_configs = []
                iter_value_envs = []
                for i, iconf in enumerate(iter_configs):
                    for param_value in param_values:
                        iter_value_config = copy.deepcopy(iconf)
                        iter_value_config[alg_param] = param_value
                        iter_value_configs.append(iter_value_config)
                        iter_value_envs.append(env)
                iter_configs.clear()
                iter_configs.extend(iter_value_configs)
                iter_envs.clear()
                iter_envs.extend(iter_value_envs)

            else:
                if len(iter_configs) == 0:
                    iter_envs.append(vars(env))
                    iter_configs.append(dict())
                for iconfig in iter_configs:
                    iconfig[alg_param] = param_values
        all_configs.extend(iter_configs)
        all_envs.extend(iter_envs)
    return all_configs, all_envs


def _save_sol(i, p, env, sols_dir, sol_dict, ts, msgs):
    sol_filename = os.path.join(sols_dir, '{:03d}_sol.json'.format(i))
    save_dict = {
        'sol': sol_dict,
        'params': _prepare_save_params(p, env),
        'info': {'sol_time': ts,
                 'check_msgs': msgs}
    }
    with open(sol_filename, 'w') as fp:
        json.dump(save_dict, fp, indent=2, cls=JSONEncoder)


def _solve(param_tuple):
    falg, fcheck, i, p, env, sols_dir = param_tuple
    print("[{}]: Start solving {:03d}".format(datetime.now().strftime("%H:%M:%S"), i))
    ts = time.time()
    _, _, sol_dict = falg(**p)
    ts = time.time() - ts
    msgs = fcheck(p, sol_dict, do_raise=False)  # TODO change for batch solving
    if msgs:
        print("!!! Warning !!!: {}".format(msgs))
    _save_sol(i, p, env, sols_dir, sol_dict, ts, msgs)
    print("[{}]: Solved {:03d} (took {:.0f}s)".format(datetime.now().strftime("%H:%M:%S"), i, ts))


def solve_batch(args):
    def _use_id(id):
        return (use_ids is None) or (id in use_ids)

    if (args.id is None) or (args.id.strip() == '') or (args.id.strip() == ':'):
        use_ids = None
    elif args.id.strip()[-1] == ':':
        use_ids = range(int(args.id.split(':')[0]), 2**32)
    elif args.id.strip()[0] == ':':
        use_ids = range(0, int(args.id.split(':')[-1]))
    else:
        use_ids = [int(s) for s in args.id.split(',')]

    scenarios_file = None
    ex_config_file = None
    envs_file = None
    if args.scenarios_dir and args.scenario_name:
        for f in os.listdir(args.scenarios_dir):
            if f == '{}.json'.format(args.scenario_name):
                scenarios_file = os.path.join(args.scenarios_dir, f)
            if f == '{}_ex_configs.json'.format(args.scenario_name):
                ex_config_file = os.path.join(args.scenarios_dir, f)
            if f == '{}_envs.json'.format(args.scenario_name):
                envs_file = os.path.join(args.scenarios_dir, f)
    else:
        scenarios_file = args.scenarios_file
        ex_config_file = args.ex_config_file
        envs_file = args.envs_file

    envs = []
    in_file_name = None
    if envs_file:
        in_file_name = os.path.splitext(os.path.split(envs_file)[-1])[0]
        with open(envs_file, 'r') as fp:
            envs = json.load(fp)
    elif scenarios_file:
        in_file_name = os.path.splitext(os.path.split(scenarios_file)[-1])[0]
        print("Info: envs will be created from scenarios file {}".format(scenarios_file))
        with open(scenarios_file, 'r') as fp:
            scenarios = json.load(fp)
            rng = np.random.RandomState(args.seed)
            envs = generate_envs(scenarios, rng, int(args.n_per_conf))

    scenario_name = args.scenario_name if args.scenario_name else in_file_name

    ex_configs = None
    if ex_config_file:
        with open(ex_config_file, 'r') as fp:
            ex_configs = json.load(fp)

    sols_dir = None
    if args.sols_dir:
        if args.create_sols_sub_dir:
            sols_dir = os.path.join(args.sols_dir, scenario_name, args.alg)
        else:
            sols_dir = args.sols_dir
        os.makedirs(sols_dir, exist_ok=True)

    envs = _prepare_env_list(envs)
    all_configs, all_envs = prepare_configs(envs, ex_configs[args.alg])
    params = _prepare_alg_params(args.alg, all_configs, all_envs, sols_dir, args)

    fcheck_sol = partial(check_solution, ignore_conn=(args.alg == 'mmccp'))
    d_alg = {
        'emicp': (emicp, -1),
        'micp_milp': (micp_milp, 1),
        'emicp_milp': (emicp_milp, 1),
        'mmccp': (_mmccp_wrapper, -1)
    }
    if args.alg not in d_alg.keys():
        raise ValueError("Alg '{}' not supported".format(args.alg))

    d_alg = d_alg[args.alg]

    if (d_alg[1] > 0) and (args.num_workers > d_alg[1]):
        raise ValueError("Alg '{}' does not support num_workers={}".format(args.alg, args.num_workers))  # it is assumed that milp algorithms use fscip which is parallelized itself

    if int(args.num_workers) > 1:
        map_f_params = [(d_alg[0], fcheck_sol, i, p, e, sols_dir) for i, (p, e) in enumerate(zip(params, all_envs)) if _use_id(i)]
        with mp.Pool(args.num_workers) as p:
            p.map(_solve, map_f_params)

    else:
        for i, (p, e) in enumerate(zip(params, all_envs)):
            if _use_id(i):
                _solve((d_alg[0], fcheck_sol, i, p, e, sols_dir))


if __name__ == '__main__':
    _parser = argparse.ArgumentParser()
    _parser.add_argument('--scenarios_dir', default='scenarios', help='Folder with scenario configurations')
    _parser.add_argument('--scenario_name', help='"small" or "obs_large"')
    _parser.add_argument('--envs_file')
    _parser.add_argument('--scenarios_file')
    _parser.add_argument('--ex_config_file')
    _parser.add_argument('--alg')
    _parser.add_argument('--sols_dir', default='scenarios/sols', help='Where the solutions will be saved')
    _parser.add_argument('--create_sols_sub_dir', action='store_true', default=False)
    _parser.add_argument('--num_workers', type=int, default=1)
    _parser.add_argument('--id', type=str, help='Which scenario variation should be solved')
    _parser.add_argument('--solver_params', type=str, default=None, help='Parameters passed to the MILP solver e.g. "TimeLimit:10800"')

    _parser.add_argument('--n_per_conf', type=int)  # if no scenario configs provided
    _parser.add_argument('--seed', type=int, default=122)  # if no scenario configs provided

    _args = _parser.parse_args(sys.argv[1:])

    solve_batch(_args)
