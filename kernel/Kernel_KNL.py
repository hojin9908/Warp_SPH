import warp as wp
import math

# To Exclude itself
R2_MIN = wp.constant(1.0e-12)

@wp.func
def Kernel_w_Wendland(r: float, h: float) -> float:
    """
    Calculate Wendland kernel value

    r: Distance between two particlese
    h: Smoothing length

    wij: Wendland kernel value
    """
    q = r / h
    wij = float(0.0)
    u = 1.0 - 0.5 * q
    if q < 2.0:
        wij = 7.0 * u * u * u * u * (1.0 + 2.0 * q) / (4.0 * math.pi * h * h) 
    return wij


@wp.func
def Kernel_dw_Wendland(r: float, h: float) -> float:
    """
    Calculate dw/dr of Wendland kernel

    r: Distance between two particles
    h: Smooting length

    dwij: dw/dr of Wendland kernel value
    """
    q = r / h
    dwij = float(0.0)
    if q < 2.0:
        u = 1.0 - 0.5 * q
        dwij = 7.0 * (-5.0 * q * u * u * u) / (4.0 * math.pi * h * h * h)
    return dwij

