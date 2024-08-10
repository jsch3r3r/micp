import sys
import os
import argparse
import json
import matplotlib.pyplot as plt

from lib.utils import show_scenario


def main(args):

    def _on_press(event):
        nonlocal index
        if len(d_envs) == 1:
            return

        if event.key == 'right':
            if index < len(d_envs) - 1:
                index += 1
                _plot_envs()
        elif event.key == 'left':
            if index > 0:
                index -= 1
                _plot_envs()

    def _get_sol(index, i):
        if not args.sol_alg:
            return None

        # TODO hard-coded stuff here is only supporting the two scenarios from the paper:
        if args.scenario_name == 'small':
            # with |V|==10 (index==0), r==1,2,3 -> args.sol_index in {0,1,2}:
            assert index in range(1)
            if args.sol_index not in range(3):
                raise ValueError
            file_index = i*3 + args.sol_index
        elif args.scenario_name == 'obs_large':
            # with |V|==10,20,...,100 (index==0,...,9), r==2,4,6,8,10 -> args.sol_index in {0,...,4}
            assert index in range(10)
            if args.sol_index not in range(5):
                raise ValueError
            file_index = index*50 + i*5 + args.sol_index
        else:
            raise ValueError

        with open(os.path.join(sol_dir, args.sol_alg, '{:03d}_sol.json'.format(file_index)), 'r') as fp:
            sol = json.load(fp)
            return sol['sol']

    def _plot_envs():
        for i, env in enumerate(d_envs[index]):
            row = i // ncols
            col = i % ncols

            _ax = ax[row, col]
            _ax.cla()
            show_scenario(env, sol=_get_sol(index, i), title=i, ax=_ax)
        fig.suptitle("{}: {}".format(args.scenario_name, index))
        fig.canvas.draw()

    scenario_dir = 'scenarios'
    sol_dir = os.path.join('scenarios/sols', args.scenario_name) if args.create_sols_sub_dir else 'scenarios/sols'

    env_file = os.path.join(scenario_dir, '{}_envs.json'.format(args.scenario_name))
    with open(env_file, 'r') as fp:
        d_envs = json.load(fp)

    nrows = 3
    ncols = 4
    index = 0

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
    fig.title = 'test'
    fig.canvas.mpl_connect('key_press_event', _on_press)
    _plot_envs()

    plt.show()


if __name__ == '__main__':
    _parser = argparse.ArgumentParser()
    _parser.add_argument('--scenario_name', help='"small" or "obs_large"')
    _parser.add_argument('--create_sols_sub_dir', action='store_true', default=False, help='If solutions should be shown as well, use this flag if it was used for solving as well')
    _parser.add_argument('--sol_alg', help='If solutions should be shown as well, which algorithm')
    _parser.add_argument('--sol_index', type=int, default=0, help='If solutions should be shown as well, which variation index')

    _args = _parser.parse_args(sys.argv[1:])

    main(_args)
