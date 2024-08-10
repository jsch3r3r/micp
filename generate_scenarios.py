import os
import argparse
import json
import sys
import numpy as np

from lib.generate_scenario import generate_envs
from lib.utils import JSONEncoder
from scenario_configs import reg


def main(args):

    try:
        _scenarios, _ex_config = reg[args.scenario_name]()
    except KeyError:
        raise ValueError("'{}' not supported".format(args.scenario_name))

    _rng = np.random.RandomState(seed=args.seed)
    _envs = generate_envs(_scenarios, _rng, args.n_per_conf)

    if args.do_save:
        os.makedirs('scenarios', exist_ok=True)
        with open(os.path.join('scenarios', '{}.json'.format(args.scenario_name)), 'w') as fp:
            json.dump(_scenarios, fp, cls=JSONEncoder, indent=2)
        with open(os.path.join('scenarios', '{}_envs.json'.format(args.scenario_name)), 'w') as fp:
            json.dump(_envs, fp, cls=JSONEncoder, indent=2)
        with open(os.path.join('scenarios', '{}_ex_configs.json'.format(args.scenario_name)), 'w') as fp:
            json.dump(_ex_config, fp, cls=JSONEncoder, indent=2)


if __name__ == '__main__':
    _parser = argparse.ArgumentParser()
    _parser.add_argument('--scenario_name', default='small', help='"small" or "obs_large" as defined in scenario_configs.reg')
    _parser.add_argument('--no_save', action='store_false', dest='do_save', default=True)
    _parser.add_argument('--n_per_conf', type=int, default=10, help='How many scenarios with randomly sampled vertices will be generatd per variation')
    _parser.add_argument('--seed', type=int, default=122, help='Seed of the random number generator')
    _args = _parser.parse_args(sys.argv[1:])

    main(_args)
