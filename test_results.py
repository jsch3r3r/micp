import argparse
import sys
import numpy as np
import os
import json
from collections import defaultdict
from functools import partial

from lib.utils import check_solution, calc_tour_len, show_scenario, JSONEncoder, remap_filtered_vertices
from lib.mmccp import mmccp
from lib.micp_milp import emicp_milp, parse_solver_out_string


parser = argparse.ArgumentParser()
parser.add_argument('--scenario_name')
parser.add_argument('--sols_dir', default='scenarios/sols')
parser.add_argument('--create_sols_sub_dir', action='store_true', default=False)
parser.add_argument('--sol_nr', type=int, default=None)
parser.add_argument('--show_single_plot', action='store_true', default=False)
parser.add_argument('--solve_single', action='store_true', default=False)
args = parser.parse_args(sys.argv[1:])

sol_nr = args.sol_nr if (args.sol_nr is not None) and (args.sol_nr > -1) else None
show_single_plot = args.show_single_plot
solve_single = args.solve_single

root_dir = os.path.join(args.sols_dir, args.scenario_name) if args.create_sols_sub_dir else args.sols_dir
algs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
algs_milp = [alg for alg in algs if alg == 'micp_milp' or alg == 'emicp_milp']
algs_real = [alg for alg in algs if alg != 'mmccp']
alg_checks = {alg: partial(check_solution, do_raise=False, ignore_conn=(alg == 'mmccp')) for alg in algs}

# sol_dir_emicp = os.path.join(root_dir, 'emicp')
# sol_dir_emicp_milp = os.path.join(root_dir, 'emicp_milp')
# sol_dir_micp_milp = os.path.join(root_dir, 'micp_milp')
sol_dirs = dict()
for alg in algs:
    sol_dirs[alg] = os.path.join(root_dir, alg)

zs = {alg: defaultdict(lambda: [-1, -1]) for alg in algs}
msgs_check = defaultdict(dict)
msgs_sol = defaultdict(dict)
sols = defaultdict(dict)
time_limits = defaultdict(dict)
solve_status = defaultdict(dict)

for alg in algs:
    for f in sorted(os.listdir(sol_dirs[alg])):
        if os.path.splitext(f)[-1] == '.json':
            with open(os.path.join(sol_dirs[alg], f)) as fp:
                df = json.load(fp)
            n = int(os.path.splitext(f)[0][:3])
            sols[alg][n] = df
            zs[alg][n][0] = float(df['sol']['z'])
            msgs_check[alg].update({n: alg_checks[alg](opts=df['params'], sol=df['sol'])})
            msgs_sol[alg].update({n: df['info']['check_msgs']})
        elif os.path.splitext(f)[-1] == '.sol':
            with open(os.path.join(sol_dirs[alg], f)) as fp:
                l = fp.readline()
            n = int(os.path.splitext(f)[0])
            try:
                zs[alg][n][1] = float(l.split(':')[1])  # first line e.g.: 'objective value:                     1.72973837022566'
            except IndexError:  # currently optimizing (incomplete file)
                print("Warning: problem with {}".format(f))
        elif os.path.splitext(f)[-1] == '.out':
            n = int(os.path.splitext(f)[0])
            solve_status[alg][n] = parse_solver_out_string(os.path.join(sol_dirs[alg], f), ['SCIP Status'])
            try:
                time_limits[alg][n] = 'hard time limit reached' in solve_status[alg][n]['SCIP Status']
            except KeyError:  # currently optimizing (incomplete file)
                print("Warning: problem with {}".format(f))


def _z_line(_n):
    s = '\t'.join([
        '{:.4f} \t {}'.format(
            zs[_alg][_n][0],
            '__' if (zs[_alg][_n][1] == -1) or (np.isclose(zs[_alg][_n][0], zs[_alg][_n][1])) else '!!'
        ) for _alg in algs
    ])
    _d = [(_alg, zs[_alg][_n][0]) for _alg in algs]
    _d = sorted(_d, key=lambda _i: _i[1])
    s += '\t\t' + ' <= '.join([_i[0] for _i in _d])

    # check known order of objective values
    _d = {_i[0]: _i[1] for _i in _d}
    _z_mmccp = _d['mmccp']
    _z_emicp = _d['emicp']
    _b = ((_z_mmccp <= _z_emicp) or np.isclose(_z_mmccp, _z_emicp))
    try:
        _z_emicp_milp = _d['emicp_milp']
        _b = _b and ((_z_mmccp <= _z_emicp_milp) or np.isclose(_z_mmccp, _z_emicp_milp)) and \
                    (True if time_limits['emicp_milp'] else ((_z_emicp_milp <= _z_emicp) or np.isclose(_z_emicp_milp, _z_emicp)))  # don't check if time limit is reached
    except KeyError:
        pass

    s += '\t' + ('__' if _b else '!!')

    return s


print('\t', '\t\t'.join(algs).upper())
for k in sols[algs[0]].keys():
    print(k, '\t', _z_line(k))

print("Total: {}".format(' / '.join(['{}: {}'.format(alg.upper(), len(sols[alg])) for alg in algs])))

for alg in algs_milp:
    print("\n")
    print("{}:".format(alg.upper()))
    ll = []
    for k, v in sols[alg].items():
        t = v['sol']['solver_info']['solve_time']
        lim = np.inf if ('solver_params' not in v['params']) or ('TimeLimit' not in v['params']['solver_params']) else\
            v['params']['solver_params']['TimeLimit']
        if float(t) >= float(lim)*0.99:
            ll.append((k, t))

    print("solver time limit: ", ["{} ({:.2f})".format(k, t) for (k, t) in ll])
    print("time limit reached: ", list(time_limits[alg].keys()))  # [k for k, _ in time_limit.items()])

for alg in algs_milp:
    print("\n")
    print("{}:".format(alg.upper()))
    for k in sols[algs[0]].keys():
        print("\t", k, solve_status[alg][k])

for alg in algs:
    print("\n")
    print("Check problems {}: {}".format(alg.upper(), {k: v for k, v in msgs_check[alg].items() if len(v) > 0}))
    print("Sol problems {}: {}".format(alg.upper(), {k: v for k, v in msgs_sol[alg].items() if len(v) > 0}))

if sol_nr is None:
    exit(0)

print("\nINFO sol {}".format(sol_nr))

f_emicp_milp = os.path.join(sol_dirs['emicp_milp'], '{:03d}_sol.json'.format(sol_nr))
f_emicp = os.path.join(sol_dirs['emicp'], '{:03d}_sol.json'.format(sol_nr))

with open(f_emicp_milp, 'r') as fp:
    sol_emicp_milp = json.load(fp)

with open(f_emicp, 'r') as fp:
    sol_emicp = json.load(fp)

print("Check sol MILP {}: ".format(sol_nr), check_solution(sol_emicp_milp['params'], sol_emicp_milp['sol'], do_raise=False))
print("Check sol EMIC {}: ".format(sol_nr), check_solution(sol_emicp['params'], sol_emicp['sol'], do_raise=False))

print("Tour lens MILP: ", [calc_tour_len(sol_emicp_milp['sol']['tours'][i], sol_emicp_milp['params']['gm']) for i in range(len(sol_emicp_milp['sol']['tours']))])
print("Tour lens EMIC: ", [calc_tour_len(sol_emicp['sol']['tours'][i], sol_emicp['params']['gm']) for i in range(len(sol_emicp['sol']['tours']))])

print("Tours MILP: ", sol_emicp_milp['sol']['tours'])
print("Tours EMIC: ", sol_emicp['sol']['tours'])

gm = np.array(sol_emicp_milp['params']['gm'])
vs = np.array(sol_emicp_milp['params']['vs'])
gm_vs = gm[:, vs][vs, :]  # filter the columns then filter the rows
tours, lens = mmccp(gm_vs, int(sol_emicp_milp['params']['r']))
tours = remap_filtered_vertices(tours, vs)
print(tours, lens)

gm = np.array(sol_emicp['params']['gm'])
vs = np.array(sol_emicp['params']['vs'])
gm_vs = gm[:, vs][vs, :]  # filter the columns then filter the rows
tours, lens = mmccp(gm_vs, int(sol_emicp['params']['r']))
tours = remap_filtered_vertices(tours, vs)
print(tours, lens)

__rnd_id__ = sol_emicp['params']['__rnd_id__']

with open('scenarios/test_envs.json', 'r') as fp:
    envs = json.load(fp)

xys = np.array(envs[0][__rnd_id__]['xys'])  # scenario '0' (only one scenario)

sol_emicp_milp['params']['xys'] = xys
sol_emicp['params']['xys'] = xys

if show_single_plot:
    show_scenario(sol_emicp_milp['params'], sol_emicp_milp['sol'], title='MILP')
    show_scenario(sol_emicp['params'], sol_emicp['sol'], title='EMIC')

if not solve_single:
    exit(0)

gc = np.zeros_like(np.array(sol_emicp_milp['params']['gc']))
# gc = gc | np.eye(len(gc), dtype=bool)

vs = np.array(sol_emicp_milp['params']['vs'])
# vs[6] = True

d_solver_params = {'TimeLimit': 3600*2}  # see scipoptsuite-7.0.2/ug/src/ug/paraParamSet.h

tours, z, sol_dict = emicp_milp(np.array(sol_emicp_milp['params']['gm']),
                                gc,
                                vs,
                                int(sol_emicp_milp['params']['b']),
                                int(sol_emicp_milp['params']['r']),
                                save_model='model/debug_model.cip',
                                save_sol='model/debug_sol.sol',
                                solver_params=d_solver_params)

with open('model/debug_sol_dict.json', 'w') as fp:
    json.dump(sol_dict, fp, indent=2, cls=JSONEncoder)

print('z: {}'.format(z))
print('tours: \n{}'.format(tours))
print('lens: {}'.format([calc_tour_len(t, gm) for t in tours]))
