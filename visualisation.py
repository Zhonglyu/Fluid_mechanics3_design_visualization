import numpy as np
def concentration(x, y, z, t,
                  m_dot, Ux, Uy, Um, DT, h):

    # 相对风速
    U_rel =  - Ux + Um

    # Galilean transform 后的下游距离 x' = x - Um*t
    x_rel = x - Um * t

    # sigma^2 (only valid for x' > 0)
    sigma2 = 2.0 * DT * x_rel / U_rel
    sigma2 = - np.where((sigma2 < 0) & (t > 0), sigma2, np.nan)

    # 横向漂移：y' = y - (Uy/U_rel) * x'
    y_rel = y + (Uy / U_rel) * x_rel

    # perfect reflector: mirror sources at z = ±h
    denom = 2.0 * sigma2
    prefactor = m_dot / (2.0 * np.pi * U_rel * sigma2)

    expo1 = - (y_rel**2 + (z - h)**2) / denom
    expo2 = - (y_rel**2 + (z + h)**2) / denom

    C = prefactor * (np.exp(expo1) + np.exp(expo2))
    return np.nan_to_num(C, nan=0.0)