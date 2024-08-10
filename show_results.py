import copy
import sys
import argparse
import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

from scenario_configs import reg as reg_scenario
from lib.micp_milp import parse_solver_out_string
from lib.utils import calc_tour_len


def _defaultdict():
    def _default():
        return defaultdict(_default)
    return defaultdict(_default)


def _from_out_file(emicp_dir, micp_dir):
    d = _defaultdict()
    efiles = [os.path.join(emicp_dir, f) for f in os.listdir(emicp_dir) if
              not os.path.isdir(os.path.join(emicp_dir, f)) and
              f.endswith('_sol.json')]

    for fesol in efiles:
        with open(fesol, 'r') as fp:
            d_esol = json.load(fp)

        sid = os.path.split(fesol)[-1][:3]
        assert int(sid) > -1
        sol_time_limit = d_esol['sol']['solver_info']['solve_time']

        fmout = os.path.join(micp_dir, '{}.out'.format(sid))
        dout = parse_solver_out_string(fmout, ['[Time]', '[Best Integer]'])

        assert len(dout['[Time]']) == len(dout['[Best Integer]'])

        time_limit_value = None
        time_limit_time = None
        for i, t in enumerate(dout['[Time]']):
            if t >= sol_time_limit:
                time_limit_value = dout['[Best Integer]'][i]
                time_limit_time = t
                break

        if time_limit_value == '-':  # take first valid value from history
            print("Warning: {}: emicp time limit {:.1f} / micp 'Time' {} / micp 'Best Integer' {}".format(sid, sol_time_limit, time_limit_time, time_limit_value))

            time_limit_value = None  # if no value in history, then take value from solution file (see below) TODO is this fine?
            for i, v in enumerate(dout['[Best Integer]']):
                try:
                    vf = float(v)
                    time_limit_value = vf
                    time_limit_time = dout['[Time]'][i]
                    break
                except ValueError:
                    pass

            print("              micp sol time {:.1f} / micp sol value {:.3f}".format(time_limit_time, time_limit_value))

        if time_limit_value is None:  # take value from solution file
            print("Warning: {}: emicp time limit {:.1f} / micp 'Time' {} / micp 'Best Integer' {}".format(sid, sol_time_limit, time_limit_time, time_limit_value))

            fmsol = os.path.join(micp_dir, os.path.split(fesol)[-1])
            with open(fmsol, 'r') as fp:
                d_msol = json.load(fp)

            time_limit_value = d_msol['sol']['z']
            time_limit_time = d_msol['sol']['solver_info']['solve_time']

            print("              micp sol time {:.1f} / micp sol value {:.3f}".format(time_limit_time, time_limit_value))

        try:
            f = float(time_limit_value)
        except ValueError:  # should not happen
            print("Warning: {}: time limit value: {}".format(sid, time_limit_value))

        k = '{}_sol.json'.format(sid)
        d[k]['sol']['z'] = time_limit_value
        d[k]['info']['sol_time'] = time_limit_time
        d[k]['params'] = copy.deepcopy(d_esol['params'])  # need params for _get_param_value()

    return d


def _save_data_values(d_grouped_data, d_inst_sols, group_by, fields, scenario_name, out_dir):
    # e.g. d_data['emicp'][1]['z'] == {'000': 1.11, ...}
    # <scenario_name>__emicp__all.dat:
    #  (sid r z sol_time)
    #  000 1 x y
    #  001 2 x y
    #  ...
    # <scenario_noame>__emicp.dat:
    #  (r z_mean z_std sol_time_mean sol_time_std)
    #  1 ...
    #  2 ...
    #  3 ...

    fname_all = '{}__{{}}__all.dat'.format(scenario_name)
    fname = '{}__{{}}.dat'.format(scenario_name)
    fname_key = '{}__{{}}_{{}}_{{}}.dat'.format(scenario_name)

    for alg, sols in d_inst_sols.items():
        lines = []
        for sid, sol in sols.items():
            line = [sid]
            for e in group_by:
                line.append(str(sol['params'][e]))
            for e in fields:
                line.append(str(sol[e[0]][e[1]]))
            lines.append(line)
        lines = sorted(lines, key=lambda item: item[0])
        header = 'C' + '\t'.join(group_by + [k[1] for k in fields])
        lines = [header] + ['\t'.join(l) for l in lines]
        with open(os.path.join(out_dir, fname_all.format(alg)), 'w') as fp:
            fp.write('\n'.join(lines))

    if len(group_by) == 1:
        # e.g. group_by == ['r']
        # d_data['emicp'][<r>]['z'] == {'000': 1.11, ...}
        for alg, data in d_grouped_data.items():
            header_keys = copy.copy(group_by)
            lines = []
            for group, gdata in data.items():
                line = [str(group)]
                for k, vs in gdata.items():
                    header_keys.extend([k + '(m)', k + '(s)'])
                    v_mean = np.mean(list(vs.values()))
                    v_std = np.std(list(vs.values()))
                    line.append(str(v_mean))
                    line.append(str(v_std))
                lines.append(line)
            lines = sorted(lines, key=lambda item: int(item[0]))  # assumes int type for groups
            header = 'C' + '\t'.join(header_keys)
            lines = [header] + ['\t'.join(l) for l in lines]
            with open(os.path.join(out_dir, fname.format(alg)), 'w') as fp:
                fp.write('\n'.join(lines))
    elif len(group_by) == 2:
        # e.g. group_by == ['n', 'r']
        # d_data['emicp'][<n>][<r>]['z'] == {'000': 1.11, ...}
        for alg, data in d_grouped_data.items():
            header_keys = copy.copy(group_by)
            lines = []
            gdata_sum = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [])))
            for group0, gdata0 in data.items():
                for group1, gdata1 in gdata0.items():
                    for k, vs in gdata1.items():
                        # actually this should be called only once for each key combination, so no empty list is required in initialization:
                        assert len(gdata_sum[group0][group1][k]) == 0
                        gdata_sum[group0][group1][k].extend(list(vs.values()))

            for group0, gdata0 in gdata_sum.items():  # 'n'
                for group1, gdata1 in gdata0.items():  # 'r'
                    for k, vs in gdata1.items():
                        header_keys.extend([k + '(m)', k + '(s)'])
                    break
                break
            header = 'C' + '\t'.join(header_keys)

            for group0, gdata0 in gdata_sum.items():  # 'n'
                lines_key0 = []
                for group1, gdata1 in gdata0.items():  # 'r'
                    line = [str(group0), str(group1)]  # 'n' 'r'
                    for k, vs in gdata1.items():
                        v_mean = np.mean(gdata_sum[group0][group1][k])
                        v_std = np.std(gdata_sum[group0][group1][k])
                        line.append(str(v_mean))
                        line.append(str(v_std))
                    lines.append(line)
                    lines_key0.append(line)
                # save order key0, key1
                lines_key0 = sorted(lines_key0, key=lambda item: int(item[1]))
                lines_key0 = [header] + ['\t'.join(l) for l in lines_key0]
                with open(os.path.join(out_dir, fname_key.format(alg, group_by[0], group0)), 'w') as fp:
                    fp.write('\n'.join(lines_key0))

            # save order key1, key0
            gdata_reversed = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [])))
            for group0, gdata0 in gdata_sum.items():
                for group1, gdata1 in gdata0.items():
                    for k, vs in gdata1.items():
                        gdata_reversed[group1][group0][k].extend(vs)
            for group0, gdata0 in gdata_reversed.items():  # 'r'
                lines_key0 = []
                for group1, gdata1 in gdata0.items():  # 'n'
                    line = [str(group1), str(group0)]  # 'n' 'r'
                    for k, vs in gdata1.items():
                        v_mean = np.mean(gdata_reversed[group0][group1][k])
                        v_std = np.std(gdata_reversed[group0][group1][k])
                        line.append(str(v_mean))
                        line.append(str(v_std))
                    lines_key0.append(line)
                    # save order key0, key1
                lines_key0 = sorted(lines_key0, key=lambda item: int(item[1]))
                lines_key0 = [header] + ['\t'.join(l) for l in lines_key0]
                with open(os.path.join(out_dir, fname_key.format(alg, group_by[1], group0)), 'w') as fp:
                    fp.write('\n'.join(lines_key0))

            # save all keys
            for i in [1, 0]:
                lines = sorted(lines, key=lambda item: int(item[i]))  # assumes int type for groups
            lines = [header] + ['\t'.join(l) for l in lines]
            with open(os.path.join(out_dir, fname.format(alg)), 'w') as fp:
                fp.write('\n'.join(lines))
    else:
        raise ValueError("More than one group_by param is not supported")  # TODO make more general, no cases


def _get_field_value(d_sol, k0, k1, d_inst_sols, alg, sid):
    try:
        return d_sol[k0][k1]
    except KeyError:
        if alg == 'mmccp':
            if k1 in ['mmccp_max_len_inc', 'mmccp_sum_len_inc']:
                v = 0.  # these are the values other algorithms are compared to, so for mmccp they are 0.
                d_sol[k0][k1] = v
                return v
        elif alg == 'emicp':
            if k1 == 'mmccp_max_len_inc':
                z_emicp = d_inst_sols['emicp'][sid]['sol']['z']
                z_mmccp = d_inst_sols['mmccp'][sid]['sol']['z']
                v = z_emicp - z_mmccp
                d_sol[k0][k1] = v
                return v
                # if np.isclose(0., z_emicp) and np.isclose(0., z_mmccp):
                #     return 1.
                # try:
                #     return z_emicp / z_mmccp
                # except ZeroDivisionError:
                #     return z_emicp  # TODO what to return?
            elif k1 == 'mmccp_sum_len_inc':
                gm = d_inst_sols['emicp'][sid]['params']['gm']
                tours_emicp = d_inst_sols['emicp'][sid]['sol']['tours']
                tours_mmccp = d_inst_sols['mmccp'][sid]['sol']['tours']
                assert len(tours_emicp) == len(tours_mmccp)
                sum_len_emicp = 0.
                sum_len_mmccp = 0.
                for t_emicp, t_mmccp in zip(tours_emicp, tours_mmccp):
                    assert set(t_mmccp) <= set(t_emicp)  # '<=' subset relation; assume that the order of tours is the same
                    sum_len_emicp += calc_tour_len(t_emicp, gm)
                    sum_len_mmccp += calc_tour_len(t_mmccp, gm)
                v = sum_len_emicp - sum_len_mmccp
                d_sol[k0][k1] = v
                return v
                # if np.isclose(0., sum_len_emicp) and np.isclose(0., sum_len_mmccp):
                #     return 1.
                # try:
                #     return sum_len_emicp / sum_len_mmccp
                # except ZeroDivisionError:
                #     return sum_len_emicp
            else:
                raise ValueError("Key '{}' not supported".format(k1))


def main(args):
    root_dir = os.path.join(args.sols_dir, args.scenario_name) if args.create_sols_sub_dir else args.sols_dir
    algs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

    if ('emicp_milp' in algs) and ('micp_milp' in algs):
        algs.append('micp_milp_time_limit')

    scenario_info = reg_scenario[args.scenario_name](True)

    fields = [('sol', 'z'), ('info', 'sol_time')]  # in which subdict (e.g. 'sol') the keys can be found (e.g. 'z')
    if ('emicp' in algs) and ('mmccp' in algs) and (len(algs) == 2):  # TODO just these two algorithms, others as well?
        fields.append(('sol', 'mmccp_max_len_inc'))
        fields.append(('sol', 'mmccp_sum_len_inc'))

    print(algs)
    print(scenario_info)

    def _get_param_value(_d_sol, _pname):
        return _d_sol['params'][_pname]

    d_grouped_data = _defaultdict()

    d_inst_sols = dict()
    for alg in algs:
        d_sols = dict()
        if alg == 'micp_milp_time_limit':
            d_sols = _from_out_file(os.path.join(root_dir, 'emicp_milp'),
                                    os.path.join(root_dir, 'micp_milp'))
        else:
            root_dir_alg = os.path.join(root_dir, alg)
            files = [os.path.join(root_dir_alg, f) for f in os.listdir(root_dir_alg) if
                     not os.path.isdir(os.path.join(root_dir_alg, f)) and
                     f.endswith('_sol.json')]  # os.path.splitext(f)[-1] == '.json']
            for f in files:
                with open(f, 'r') as fp:
                    d_sols[os.path.split(f)[-1]] = json.load(fp)

        d_inst_sols[alg] = dict()
        for f, d_sol in d_sols.items():
            sid = f[:3]
            d_inst_sols[alg][sid] = d_sol

    for alg in algs:
        for sid, d_sol in d_inst_sols[alg].items():
            d = d_grouped_data[alg]
            for p in scenario_info['group_by'][:-1]:
                v = _get_param_value(d_sol, p)
                d = d[v]
            param_value = _get_param_value(d_sol, scenario_info['group_by'][-1])

            field_values = dict()
            for k0, k1 in fields:
                field_values[k1] = _get_field_value(d_sol, k0, k1, d_inst_sols, alg, sid)
            # z = d_sol['sol']['z']
            # sol_time = d_sol['info']['sol_time']

            for k, data_value in field_values.items():
                if d[param_value].get(k, None) is None:
                    d[param_value][k] = {sid: data_value}
                else:
                    d[param_value][k].update({sid: data_value})

    print('Data:')
    for k, v in d_grouped_data.items():
        print("{}: {}".format(k, v))

    if args.data_out_dir:
        os.makedirs(args.data_out_dir, exist_ok=True)
        _save_data_values(d_grouped_data, d_inst_sols, scenario_info['group_by'], fields,
                          args.scenario_name, args.data_out_dir)

    if len(scenario_info['group_by']) == 1:
        fig, ax = plt.subplots(layout='constrained')
        ax.set_xlabel('Number of robots')
        ax.set_ylabel('z')
        width = 0.1
        for i, (alg_name, alg_values) in enumerate(d_grouped_data.items()):
            data = np.array([[x, np.mean(list(y['z'].values()))] for x, y in alg_values.items()])
            ax.bar(data[:, 0] + width*i, data[:, 1], width, label=alg_name)
            if i == 0:
                ax.set_xticks(np.arange(1.2, max(data[:, 0]) + 1.2), labels=range(1, int(max(data[:, 0]) + 1)))

        ax.legend()
        plt.show()


if __name__ == '__main__':
    _parser = argparse.ArgumentParser()
    _parser.add_argument('--scenario_name', help='"small" or "obs_large"')
    _parser.add_argument('--sols_dir', default='scenarios/sols', help='Folder containing the solutions')
    _parser.add_argument('--create_sols_sub_dir', action='store_true', default=False)
    _parser.add_argument('--data_out_dir', default=None, help='Folder to write the results')

    _args = _parser.parse_args(sys.argv[1:])

    main(_args)
