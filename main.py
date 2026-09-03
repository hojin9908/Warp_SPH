import argparse
from typing import Any

import numpy as np
import warp as wp

from source.Config import Solv
from source.struct import SPHptl, BNDptl
from source.gen_ptl import DamPtlGeneration
from source.Simulation import SPH_OneStep


def parsing() -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Warp SPH Solver")

    parser.add_argument("--device", type=str, help="device to use", default="cuda:0")

    args = vars(parser.parse_args())
    return args


def run_forward(solv: Solv, P_sph: SPHptl, P_bnd: BNDptl) -> None:
    """
    Run Forward SPH Simulation

    solv: configuration of solv file
    P_sph: Particle structure of SPH particles      [N_sph]
    P_bnd: Particle structure of BND particles      [N_bnd] 
    """
    print(f"\n[forward] {solv.n_steps} step \t\t (t={solv.n_steps*solv.dt:.3f} s\tdt={solv.dt:.1e} s)\n")
    for _ in range(solv.n_steps):
        SPH_OneStep(solv.dt, P_sph, P_bnd)
    



def main() -> None:
    args = parsing()
    solv = Solv(**args)
    dam_generater = DamPtlGeneration(solv=solv)

    P_sph, P_bnd = dam_generater.build()            # [N_sph, SPHptl], [N_bnd, BNDptl]
