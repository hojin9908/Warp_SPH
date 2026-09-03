import warp as wp

from source.Config import Solv
from source.struct import SPHptl, BNDptl

def SPH_OneStep(dt: float, P_sph: SPHptl, P_bnd: BNDptl):

