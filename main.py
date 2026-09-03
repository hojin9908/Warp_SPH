import argparse
from typing import Any

import numpy as np
import warp as wp

from source.Config import Solv
from source.struct import SPHptl, BNDptl
from source.gen_ptl import DamPtlGeneration


def parsing() -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Warp SPH Solver")

    parser.add_argument("--device", type=str, help="device to use", default="cuda:0")

    args = vars(parser.parse_args())
    return args


def run_forward(solv: Solv, P_sph: SPHptl, P_bnd: BNDptl):



def main() -> None:
    args = parsing()
    solv = Solv(**args)
    dam_generater = DamPtlGeneration(solv=solv)

    P_sph, P_bnd = dam_generater.build()            # [N_sph, SPHptl], [N_bnd, BNDptl]
