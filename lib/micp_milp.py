import sys
import os
import numpy as np
from typing import Any, Union, Tuple
from datetime import datetime
import time
import json
from subprocess import Popen, PIPE
from pyscipopt import Model
from pyscipopt.scip import Solution

from .mmccp import mmccp, check_mmccp_sol
from .utils import remap_filtered_vertices


# PATH_TO_FSCIP = '/opt/scipoptsuite-7.0.2-ug/ug/bin/fscip'

FILENAME_FSCIP_PARAMS = 'fscip_params.par'
FILENAME_SOL = 'model.sol'  # default path where fscip saves the solution
FILENAME_OUT = 'scip_out_{}.out'  # stdout is saved to this file
use_fscip = True

STR_TO_OP = {'==': 0, '<=': 1, '>=': 2, '<': 3, '>': 4}
STR_TO_RED = {'no': -1, 'sum': 0}


def _generate_model(gm: np.ndarray, gc: np.ndarray, vs: np.ndarray, b: int, r: int, tours=None) -> tuple[Model, dict]:
    """

    :param gm: weighted adjacency matrix of complete movement graph
    :param gc: 0/1 adjacency matrix of connectivity graph
    :param vs: 0/1 vector of sensing location indices
    :param b: base station vertex
    :param r: number of robots
    :return:
    """

    def init_matrix(mat: np.ndarray, name: str, vtype: str):
        var_dict[name] = {'shape': mat.shape, 'vtype': vtype}
        with np.nditer(mat, op_flags=['writeonly'], flags=['multi_index', 'refs_ok']) as it:
            for e in it:
                e[...] = model.addVar('{}{}'.format(name, it.multi_index).replace(' ', ''), vtype=vtype)  # the solution cannot be read if the variable names contain spaces
        return mat

    def add_var(name: str, vtype: str):
        assert name.find(' ') < 0, "Varible names may not contain spaces"  # the solution cannot be read if the variable names contain spaces
        var_dict[name] = {'shape': 1, 'vtype': vtype}
        return model.addVar(name, vtype=vtype)

    def add_cons_single(lhs, rhs, iop: int):
        if iop == 0:  # '=='
            model.addCons(lhs == rhs)
        elif iop == 1:  # '<='
            model.addCons(lhs <= rhs)
        elif iop == 2:  # '>='
            model.addCons(lhs >= rhs)
        elif iop == 3:  # '<'
            model.addCons(lhs < rhs)
        elif iop == 4:  # '>'
            model.addCons(lhs > rhs)
        else:
            raise ValueError

    def reduce_var(v1, v2, ired):
        if ired == 0:  # 'sum'
            return v1 + v2
        else:
            raise NotImplementedError

    def reduce_vars_masked(variables: np.ndarray, mask: np.ndarray, reduce_op='no'):
        return add_cons_masked(variables=variables, mask=mask, op='==', rhs_value=0, reduce_op=reduce_op, omit=True)

    def add_cons_masked(variables: np.ndarray, mask: Union[np.ndarray, None], op: str, rhs_value: Any,
                        reduce_op='no', omit=False):
        iop = STR_TO_OP[op]
        ired = STR_TO_RED[reduce_op]
        if mask is None:
            mask = np.ones_like(variables, dtype=bool)
        if isinstance(rhs_value, np.ndarray):
            assert (variables.shape == rhs_value.shape) or (np.prod(rhs_value.shape) == 1)
        red_res = None
        if ired >= 0:
            red_res = model.addVar('', vtype='C')  # TODO
            model.addCons(red_res == 0)  # this is the initial variable for recurrent variable reduction, e.g. red_res = red_res + e
        with np.nditer(variables, op_flags=['readonly'], flags=['multi_index', 'refs_ok']) as it:
            for e in it:
                if mask[it.multi_index]:
                    if ired >= 0:
                        red_res = reduce_var(red_res, e.item(), ired)
                    else:
                        if isinstance(rhs_value, np.ndarray):
                            rhs = rhs_value[it.multi_index]
                        else:
                            rhs = rhs_value
                        add_cons_single(e.item(), rhs, iop)  # .item() is required because type(e)==np.ndarray
        if ired >= 0:
            if not omit:
                add_cons_single(red_res, rhs_value, iop)
            return red_res
        return None

    assert not vs[b], "BS must not be SL"

    vs = vs.astype(bool)
    gc = gc.astype(bool)

    n = gm.shape[0]
    mc = n * gm.max()
    assert mc < np.inf, "gm must be complete graph."

    gme = np.concatenate([gm, gm], axis=1)  # G_m G_m
    model = Model('MICP')

    var_dict = {}

    x = np.zeros((2*n, 2*n, r), dtype=object)
    xc = np.zeros((2*n, 2*n), dtype=object)
    u = np.zeros((2*n, r), dtype=object)
    f = np.zeros((2*n, 2*n, n), dtype=object)

    z_r = np.zeros((r, ), dtype=object)

    init_matrix(x, 'x', 'B')
    init_matrix(xc, 'xc', 'B')
    init_matrix(u, 'u', 'C')
    init_matrix(f, 'f', 'C')
    init_matrix(z_r, 'z_r', 'C')

    z = add_var('z', 'C')

    # excluded variables
    # | I 0|  |n-to-n, n-to-a|, no self-edges (i-to-i' must be kept for single vertex tours)
    # |~I 1|  |a-to-n, a-to-a|, only from artificial to corresponding normal, no artificial-to-artificial
    v = np.concatenate([np.concatenate([np.eye(n, dtype=bool), np.zeros((n, n), dtype=bool)], axis=1),
                        np.concatenate([~np.eye(n, dtype=bool), np.ones((n, n), dtype=bool)], axis=1)],
                       axis=0).astype(bool)  # (2*n, 2*n)

    v = np.repeat(np.expand_dims(v, axis=2), r, axis=2)  # (2*n, 2*n, r), r times repeated in the most inner dim

    add_cons_masked(x, v, '==', 0)  # model.addCons(x[v] == 0) does not work

    # at most one artificial vertex for each robot
    #  |0 0|
    #  |I 0|
    v = np.concatenate([np.concatenate([np.zeros((n, n), dtype=bool), np.zeros((n, n), dtype=bool)], axis=1),
                        np.concatenate([np.eye(n, dtype=bool), np.zeros((n, n), dtype=bool)], axis=1)],
                       axis=0).astype(bool)  # (2*n, 2*n)

    # sum_{i=n...2*n-1, j=i-n)}{x[i, j, k]} <= 1, foreach k
    for k in range(r):  # sum variables up for each k
        add_cons_masked(x[..., k], v, '<=', 1, reduce_op='sum')

    # node potentials
    for k in range(r):
        for i in range(n):
            for j in range(2*n):
                if j != i+n:
                    # x[i, j, k] == 1 => (u[i, k] + gme[i, j] <= u[j, k])
                    # x[i, j, k] == 0 => (u[i, k] - u[j, k] + gme[i, j] <= M), constraint disabled
                    # (i, j) in
                    #  1 ~I
                    #  0  0
                    # (n-to-n/a)
                    model.addCons(u[i, k] - u[j, k] + mc*x[i, j, k] <= mc - gme[i, j])

    # sensing locations: each exactly once (why not at least one?)
    for i in range(n):
        if vs[i]:
            # j in
            #  [1 ... 1 0 1 ... 1 0 ... 0 1 0 ... 0]^T for each k
            #           i                i+n
            # (n(~i)/a(i)-to-n(i))
            v1 = np.concatenate([np.ones((n,), dtype=bool), np.zeros((n,), dtype=bool)], axis=0)  # (2*n,)
            v1[i] = False
            v1[i + n] = True
            v1 = np.repeat(np.expand_dims(v1, axis=1), r, axis=1)  # (2*n, r)

            # sum_{j in n(~i)/a(i), k}{x[j, i, k]} == 1
            add_cons_masked(x[:, i, :], v1, '>=', 1, reduce_op='sum')  # x[:, i, :].shape==(2*n, r)

    # sensing locations: at most once for particular k, in-deg == out-deg
    for k in range(r):
        for i in range(n):
            if vs[i]:
                # j in
                #  [1 ... 1 0 1 ... 1 0 ... 0 1 0 ... 0]^T
                #           i                i+n
                # (n(~i)/a(i)-to-n(i))
                v1 = np.concatenate([np.ones((n,), dtype=bool), np.zeros((n,), dtype=bool)], axis=0)  # (2*n,)
                v1[i] = False
                v1[i + n] = True

                #  [1 ... 1 0 1 ... 1 1 ... 1 1 1 ... 1]^T
                #           i                i+n
                # (n(~i)/a-to-n(i))
                v2 = np.concatenate([np.ones((n,), dtype=bool), np.ones((n,), dtype=bool)], axis=0)  # (2*n,)
                v2[i] = False

                # sum_{j in n(~i)/a(i)}{x[j, i, k]} <= 1
                add_cons_masked(x[:, i, k], v1, '<=', 1, reduce_op='sum')  # x[:, i, k].shape==(2*n,)
                # sum_{j in n(~i)/a(i)}{x[j, i, k]} == sum_{j in n(~i)/a}{x[i, j, k]}
                lhs = reduce_vars_masked(x[:, i, k], v1, 'sum')  # x[:, i, k].shape==(2*n,)
                rhs = reduce_vars_masked(x[i, :, k], v2, 'sum')  # x[i, :, k].shape==(2*n,)
                model.addCons(lhs == rhs)

    # additional vertices: in-deg == out-deg
    for k in range(r):
        for i in range(n):
            if not vs[i]:
                v1 = np.concatenate([np.ones((n,), dtype=bool), np.zeros((n,), dtype=bool)], axis=0)  # (2*n,)
                v1[i] = False
                v1[i + n] = True

                v2 = np.concatenate([np.ones((n,), dtype=bool), np.ones((n,), dtype=bool)], axis=0)  # (2*n,)
                v2[i] = False
                v2[i + n] = False  # why?

                lhs = reduce_vars_masked(x[:, i, k], v1, 'sum')  # x[:, i, k].shape==(2*n,)
                rhs = reduce_vars_masked(x[i, :, k], v2, 'sum')  # x[i, :, k].shape==(2*n,)
                model.addCons(lhs == rhs)

    # artificial vertices: at most one sum of in-deg == out-deg
    for k in range(r):
        for i in range(n):
            v = np.concatenate([np.ones((n,), dtype=bool), np.zeros((n,), dtype=bool)], axis=0)  # (2*n,)
            add_cons_masked(x[:, i+n, k], v, '==', x[i+n, i, k], reduce_op='sum')  # x[:, i+n, k].shape==(2*n,)

    # given tour orders (solution of MMCCP):
    if tours is not None:
        assert check_mmccp_sol(tours, vs, r)

        # order the node potentials to enforce given vertex order within the tour and enforce the single artificial
        #  vertex edge
        for k in range(r):
            if k < len(tours):
                for ind_i, i in enumerate(tours[k]):
                    ind_j = (ind_i + 1) % len(tours[k])
                    j = tours[k][ind_j]
                    if ind_j == 0:  # j is first vertex on tour again
                        # any vertex on the tour can be the artificial vertex, force it to be the first one:
                        model.addCons(u[j, k] == 0)
                        model.addCons(u[i, k] + gm[i, j] <= u[j+n, k])  # u_i <= u_j', gm[i, i] should be 0 (in case len(tours[k]) == 1)
                        model.addCons(x[j+n, j, k] == 1)  # x_j'j == 1
                    else:
                        model.addCons(u[i, k] + gm[i, j] <= u[j, k])
                        model.addCons(x[i, j+n, k] == 0)  # x_ij' == 0
                    model.addCons(x[:, i, k].sum() == 1)  # ensures that vertex i gets visited by robot k (moves into vertex)
                    if len(tours[k]) > 1:  # i-to-i' must be possible for single vertex tours
                        model.addCons(x[i, i+n, k] == 0)  # x_ii' == 0
            else:
                # TODO use mask argument to slice 'k:'
                add_cons_masked(x[:, :, k], None, '==', 0)
                add_cons_masked(u[:, k], None, '==', 0)
                model.addCons(z_r[k] == 0)  # add_cons_masked(z_r[k:], None, '==', 0)
                break

    # connectivity
    if gc is not None:
        # connectivity edges: no edge when no edge in Gc (force unused edges to zero)
        v = np.concatenate([np.concatenate([~gc, ~(gc | np.eye(n, dtype=bool))], axis=1),
                            np.concatenate([~np.eye(n, dtype=bool), np.ones((n, n), dtype=bool)], axis=1)],
                           axis=0)  # (2*n, 2*n)
        add_cons_masked(xc, v, '==', 0)

        # connectivity edges: not both path and conn (if conn => no path)
        # xc==1 => sum(x, 2)==0
        add_cons_masked(np.sum(x, axis=2), None, '<=', r*(1 - xc))  # np.sum(x, axis=2).shape==xc.shape==(2*n, 2*n)

        # connectivity edges: no comm if no path (no path in-edge => no comm out-edge)
        for i in range(2*n):
            model.addCons(np.sum(x[:, i, :]) >= np.sum(xc[i, :]))  # sum(sum(x(:, i, :)))==0 => sum(xc(i, :))==0

        # flow: only flow from sensing locations (there is no commodity for non-SLs)
        if not np.all(vs):
            add_cons_masked(f[:, :, ~vs], None, '==', 0)

        # flow: out, in flow sensing locations and BS
        for c in range(n):
            if vs[c]:
                if c != b:  # can b be SL?
                    model.addCons(np.sum(f[:, b, c]) - np.sum(f[b, :, c]) == 1)  # BS is sink of flow for each SL
                    model.addCons(np.sum(f[:, c, c]) - np.sum(f[c, :, c]) == -1)  # each SL c is source of flow type c
                    add_cons_masked(f[:, :, c], None, '<=', np.sum(x, axis=2) + xc)  # f[:, :, c].shape == (2*n, 2*n)

                for i in range(2*n):
                    if (i != b) and (i != c):
                        model.addCons(np.sum(f[:, i, c]) - np.sum(f[i, :, c]) == 0)

        add_cons_masked(f, None, '>=', 0)
        add_cons_masked(f, None, '<=', 1)

    # objective
    for k in range(r):
        add_cons_masked(u[:, k], None, '<=', z_r[k])

    add_cons_masked(z_r, None, '<=', z)
    add_cons_masked(u, None, '>=', 0)

    model.setObjective(z)

    # You can't add multiple constraints in an np.ndarray, this does not work:
    # c1 = x[0, 0, 0] == 0
    # c2 = x[0, 1, 0] == 0
    # print(type(c1))
    # model.addCons(np.array([c1, c2], dtype=object))  # 'AssertionError: given constraint is not ExprCons but ndarray'

    return model, var_dict


def _generate_sol_dict(model: Model, sol: Solution, var_dict: dict) -> dict:
    # sol[[v for v in model.getVars() if v.name[:3] == 'z_r'][1]] -> float
    sol_dict = dict()
    for var_name, var_info in var_dict.items():
        var_shape = var_info['shape']
        var_type = var_info['vtype']
        if isinstance(var_shape, tuple):  # np.prod(var_shape) > 1:
            name_vars = [v for v in model.getVars() if v.name[:len(var_name) + 1] == (var_name + '(')]
            mat = np.zeros(var_shape, dtype=float if var_type == 'C' else int)  # TODO just all float?
            with np.nditer(mat, op_flags=['writeonly'], flags=['multi_index', 'refs_ok']) as it:
                for e in it:
                    v = [v for v in name_vars if v.name == '{}{}'.format(var_name, it.multi_index).replace(' ', '')][0]
                    e[...] = sol[v]
            sol_dict[var_name] = mat
        else:
            name_vars = [v for v in model.getVars() if v.name == var_name]
            sol_dict[var_name] = sol[name_vars[0]]
    return sol_dict


def _solution_from_sol_dict(d: dict) -> Tuple[list, float]:
    x = d['x']  # (2*n, 2*n, r)
    n = int(x.shape[0] / 2)
    r = x.shape[-1]
    z = d['z']
    tours = []
    for k in range(r):
        tour = []
        xk = x[..., k]
        vis = np.zeros(2*n, dtype=bool)  # including the artificial vertices
        v = 0
        while True:
            try:
                vnew = xk[v, :].nonzero()[0][0]  # (1)
                if vis[v]:
                    raise IndexError  # (2)
                else:
                    tour.append(v)
                    vis[v] = True
                v = vnew
            except IndexError:  # (1) v is not part of a cycle, or (2) cycle is finished
                vis[v] = True
                try:
                    v = (~vis).nonzero()[0][0]  # find next starting vertex of a cycle
                except IndexError:  # no unvisited vertex left
                    break

        tour_red = [v for v in tour if v < n]  # removes artificial vertices
        tours.append(tour_red)

    return tours, z


def _solve(model, filename_model, filename_sol=None, params_dict=None):
    if use_fscip:
        with open('options.json', 'r') as fp:
            _d_options = json.load(fp)
        PATH_TO_FSCIP = _d_options['path_to_fscip']

        if params_dict:
            with open(FILENAME_FSCIP_PARAMS, 'w') as fp:
                for k, v in params_dict.items():
                    fp.write('{}={}\n'.format(k, v))
        if filename_sol:
            assert os.path.splitext(filename_sol)[-1] == '.sol'  # this is a safety measure
            if os.path.exists(filename_sol):
                os.remove(filename_sol)
        params = [PATH_TO_FSCIP,
                  FILENAME_FSCIP_PARAMS,
                  filename_model,
                  '-fsol',
                  (filename_sol if filename_sol else FILENAME_SOL)]
        process = Popen(params, stdout=PIPE, stderr=PIPE)
        while process.returncode is None:
            for line in process.stdout:
                print(line)
            process.poll()
        model.readSol(filename_sol if filename_sol else FILENAME_SOL)  # TODO where does fscip save the solution?
    else:
        model.optimize()
        model.writeSol(model.getBestSol(), filename_sol if filename_sol else FILENAME_SOL)  # TODO is this the same file that fscip outputs by default ('model.sol')?


def parse_solver_out_string(filename: str, keys: list = None):
    if keys is None:
        keys = ['SCIP Status', 'Gap', 'Total Time']  # '[Time]', '[Best Integer]'

    parse_history = len([1 for k in keys if k.startswith('[')]) > 0
    parse_history_started = False
    parse_history_stopped = False
    d_history = None
    if parse_history:
        history_header = ['Time', 'Nodes', 'Nodes Left', 'Active Solvers', 'Best Integer', 'Best Node', 'Gap',
                          'Best Node(S)', 'Gap(S)']
        d_history = dict()
        for i, h in enumerate(history_header):
            if '[' + h + ']' in keys:
                d_history[h] = (i, [])

    def _parse_line(line):
        _l = line.split()  # splits on whitespaces and other separators, e.g. '\n' (ignoring multiple consecutive)
        for k, (_ind, _vs) in d_history.items():
            if _ind < len(_l):
                _sv = _l[_ind]
                if _sv.endswith('%'):
                    _sv = _sv[:-1]
                try:
                    _v = float(_sv)
                except ValueError:
                    _v = _sv
                _vs.append(_v)
            else:
                _vs.append('')

    d = dict()
    with open(filename, 'r') as fp:
        sl = fp.readline()
        while sl:
            sl = sl.strip()
            if sl.startswith("b'"):
                sl = sl[2:-3].strip()

            if parse_history and parse_history_started and (not parse_history_stopped):
                if sl.startswith('*'):
                    sl = sl[1:].strip()
                if not sl.startswith('Racing ramp-up'):  # e.g. 'Racing ramp-up finished after 17.3900 seconds. Selected strategy 7.'
                    if sl[0].isdigit():  # assumes that first column is 'Time' and always contains a number
                        _parse_line(sl)
                    else:
                        parse_history_stopped = True

            for k in keys:
                if sl.strip().startswith(k):
                    vsl = sl.split(':')[1].strip()
                    d[k] = vsl
                    if parse_history_started:
                        parse_history_stopped = True

            if parse_history and (not parse_history_started) and sl.strip().startswith('Time'):
                parse_history_started = True

            sl = fp.readline()

    if parse_history:
        for k, (_, vs) in d_history.items():
            d['[' + k + ']'] = vs

    return d


def micp_milp(gm: np.ndarray, gc: np.ndarray, vs: np.ndarray, b: int, r: int, save_model='model/model.cip', load_sol=None,
              save_sol='model/model.sol', tours=None, solver_params=None):
    """
    Solve MILP formulation of MICP (minimum idleness connectivity-constrained patrolling).
    :param gm: Undirected complete graph with travel distances on the edges (inf: no path).
    :param gc: Undirected 0/1 graph with edges indicating connectivity (0: disconnected, 1: connected).
    :param vs: List of sensing location indices in gm/gc
    :param b: Base station index in gm/gc
    :param r: Number of robots (cycles)
    :param save_model:
    :param load_sol:
    :param save_sol:
    :param tours:
    :param solver_params: values written to fscip argument 'fscip_param_file' (fscip_params.par)
    :return:
    """
    model_time = time.time()
    model, var_dict = _generate_model(gm, gc, vs, b, r, tours)
    model_time = time.time() - model_time
    if not save_model:
        save_model = 'model.cip'
    model.writeProblem(save_model)  # writes to 'model.cip' if no filename is provided -> read from solver

    filename_out = None
    solve_time = time.time()
    if load_sol:
        model.readSol(load_sol)
    else:
        filename_out = os.path.splitext(save_sol)[0] + '.out' if save_sol else \
            FILENAME_OUT.format(datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
        stdout = sys.stdout
        with open(filename_out, 'w') as sys.stdout:
            _solve(model, save_model, save_sol, solver_params)
        sys.stdout = stdout  # the original value must be restored, otherwise subsequent print() cause 'IOError' (file is closed)
    solve_time = time.time() - solve_time

    sol = model.getBestSol()

    if save_sol:
        model.writeSol(sol, save_sol)

    sol_dict = _generate_sol_dict(model=model, sol=sol, var_dict=var_dict)  # sol contains variable 'z'
    tours, z = _solution_from_sol_dict(sol_dict)
    sol_dict['tours'] = tours
    sol_dict['solver_info'] = {'solve_time': solve_time,
                               'model_time': model_time}
    if filename_out is not None:
        sol_dict['solver_info'].update({'status': parse_solver_out_string(filename_out)})

    return tours, z, sol_dict


def emicp_milp(gm: np.ndarray, gc: np.ndarray, vs: np.ndarray, b: int, r: int, save_model='model/model.cip', load_sol=None,
               save_sol='model/model.sol', solver_params=None):
    """
    Solve MMCP with heuristic, and e-MICP with MILP model.
    :param gm:
    :param gc:
    :param vs:
    :param b:
    :param r:
    :param save_model:
    :param load_sol:
    :param save_sol:
    :param solver_params:
    :return:
    """
    gm_vs = gm[:, vs][vs, :]  # filter the columns then filter the rows
    tours, _ = mmccp(gm_vs, r)
    tours = remap_filtered_vertices(tours, vs)
    return micp_milp(gm, gc, vs, b, r, save_model, load_sol, save_sol, tours, solver_params)
