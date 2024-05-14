from pathlib import Path
from smarts.sstudio import gen_scenario
from smarts.sstudio.types import (
    Mission,
    Route,
    Scenario,
)

gen_scenario(
    scenario=Scenario(
        traffic=None,
        ego_missions=[
            Mission(Route(begin=("edge-east-EW", 0, 10), end=("edge-west-EW", 0, 60))),
            Mission(Route(begin=("edge-south-SN", 0, 10), end=("edge-north-SN", 0, 60))),
            Mission(Route(begin=("edge-north-NS", 0, 10), end=("edge-south-NS", 0, 60))),
            Mission(Route(begin=("edge-west-WE", 0, 10), end=("edge-north-SN", 0, 60))),
        ],
    ),
    output_dir=Path(__file__).parent,
)
