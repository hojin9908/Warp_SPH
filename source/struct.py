import warp as wp

@wp.struct
class SPHptl:
    """
    - Structure of SPH (fluid) particles.
    - SoA structure
    - Read with "P.pos[i]"
    """
    pos: wp.array(dtype=wp.vec3)        # [N_sph]   Position       [m]
    vel: wp.array(dtype=wp.vec3)        # [N_sph]   Velocity       [m/s]
    rho_raw: wp.array(dtype=float)      # [N_sph]   Density (direct summation)     [kg/m^3]
    rho: wp.array(dtype=float)          # [N_sph]   Density (after Shepard filter) [kg/m^3]
    pres: wp.array(dtype=float)         # [N_sph]   Pressure       [Pa]
    acc: wp.array(dtype=wp.vec3)        # [N_sph]   Acceleration   [m/s^2]
    p_type: wp.array(dtype=int)         # [N_sph]   Particle type  (always 1)

@wp.struct
class BNDptl:
    """
    - Structure of boundary (dummy) particles.
    - SoA structure
    - Read with "B.pos[i]"
    """
    pos: wp.array(dtype=wp.vec3)        # [N_bnd]   Position       [m]
    vel: wp.array(dtype=wp.vec3)        # [N_bnd]   Velocity       [m/s]
    rho_raw: wp.array(dtype=float)      # [N_bnd]   Density (direct summation)     [kg/m^3]
    rho: wp.array(dtype=float)          # [N_bnd]   Density (after Shepard filter) [kg/m^3]
    pres: wp.array(dtype=float)         # [N_bnd]   Pressure       [Pa]
    acc: wp.array(dtype=wp.vec3)        # [N_bnd]   Acceleration   [m/s^2]
    p_type: wp.array(dtype=int)         # [N_bnd]   Particle type  (always 0)
