from .small import generate_scenarios_small
from .obs_large import generate_scenario_obs_large


reg = {'small': generate_scenarios_small,
       'obs_large': generate_scenario_obs_large
       }
