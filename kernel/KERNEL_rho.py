import warp as wp

from kernel.KERNEL_KNL import Kernel_w_Wendland, R2_MIN
from source.struct import SPHptl, BNDptl

@wp.kernel
def Kernel_shepard_sph(P_sph: SPHptl,
                       P_bnd: BNDptl,
                       grid_sph: wp.uint64,
                       grid_bnd: wp.uint64,
                       support: float,
                       h: float) -> None:
    """
    Calculate Shepard filter for SPH from SPH, BND
    
    # Output
    P_sph.flt[i]
    """
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid_sph, tid)
    ri = P_sph.pos[i]
    flt = float(0.0)
    # Sph - Sph
    for j in wp.hash_grid_query(grid_sph, ri, support):
        rij = ri - P_sph.pos[j]
        tdist = wp.dot(rij, rij)
        if tdist < support * support:
            wij = Kernel_w_Wendland(wp.sqrt(tdist), h)
            flt = flt + (P_sph.m[j] / P_sph.rho[j]) * wij
    # Sph - Bnd
    for j in wp.hash_grid_query(grid_bnd, ri, support):
        rij = ri - P_bnd.pos[j]
        tdist = wp.dot(rij, rij)
        if tdist < support * support:
            wij = Kernel_w_Wendland(wp.sqrt(tdist), h)
            flt = flt + (P_bnd.m[j] / P_bnd.rho[j]) * wij
    P_sph.flt[i] = flt




@wp.kernel
def Kernel_density_sph(P_sph: SPHptl,
                       P_bnd: BNDptl,
                       grid_sph: wp.uint64,
                       grid_bnd: wp.uint64,
                       support: float,
                       h: float) -> None:
    """
    Calculate density of SPH from SPH, BND

    The kernel sum runs over the same neighbours as Kernel_shepard_sph, so that
    the sum and the filter P_sph.flt it is divided by stay consistent.

    # Output
    P_sph.rho[i]
    """
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid_sph, tid)
    ri = P_sph.pos[i]
    rhoi = float(0.0)
    # Sph - Sph
    for j in wp.hash_grid_query(grid_sph, ri, support):
        rij = ri - P_sph.pos[j]
        tdist = wp.dot(rij, rij)
        if tdist < support * support:
            wij = Kernel_w_Wendland(wp.sqrt(tdist), h)
            rhoi = rhoi + P_sph.m[j] * wij
    # Sph - Bnd
    for j in wp.hash_grid_query(grid_bnd, ri, support):
        rij = ri - P_bnd.pos[j]
        tdist = wp.dot(rij, rij)
        if tdist < support * support:
            wij = Kernel_w_Wendland(wp.sqrt(tdist), h)
            rhoi = rhoi + P_bnd.m[j] * wij
    P_sph.rho[i] = rhoi / P_sph.flt[i]




@wp.kernel
def Kernel_density_bnd(P_sph: SPHptl,
                       P_bnd: BNDptl,
                       grid_sph: wp.uint64,
                       grid_bnd: wp.uint64,
                       support: float,
                       h: float) -> None:
    """
    Calculate density of BND from SPH, BND

    Direct summation without the Shepard filter. Normalizing a wall particle
    would drag its density back to rho0 and erase the compression that the
    approaching fluid causes, which is exactly what makes the wall push back.

    # Output
    P_bnd.rho[i]
    """
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid_bnd, tid)
    ri = P_bnd.pos[i]
    rhoi = float(0.0)
    # Bnd - Sph
    for j in wp.hash_grid_query(grid_sph, ri, support):
        rij = ri - P_sph.pos[j]
        tdist = wp.dot(rij, rij)
        if tdist < support * support:
            wij = Kernel_w_Wendland(wp.sqrt(tdist), h)
            rhoi = rhoi + P_sph.m[j] * wij
    # Bnd - Bnd
    for j in wp.hash_grid_query(grid_bnd, ri, support):
        rij = ri - P_bnd.pos[j]
        tdist = wp.dot(rij, rij)
        if tdist < support * support:
            wij = Kernel_w_Wendland(wp.sqrt(tdist), h)
            rhoi = rhoi + P_bnd.m[j] * wij
    P_bnd.rho[i] = rhoi




