import pickle
from pathlib import Path
from smarts.sstudio import gen_scenario
from smarts.sstudio.types import (
    Mission,
    Route,
    Scenario,
)

with open('routes.pkl', 'rb') as f:
    # Load the content of the file using pickle.load()
    routes = pickle.load(f)

gen_scenario(
    scenario=Scenario(
        traffic=None,
        ego_missions=[
            Mission(Route(begin=("edge-east-EW", 0, 10), end=(routes[78][0], 0, 60))),
            Mission(Route(begin=("edge-west-WE", 0, 10), end=(routes[78][1], 0, 60))),
            Mission(Route(begin=("edge-north-NS", 0, 10), end=(routes[78][2], 0, 60))),
            Mission(Route(begin=("edge-south-SN", 0, 10), end=(routes[78][3], 0, 60))),
        ],
    ),
    output_dir=Path(__file__).parent,
)
