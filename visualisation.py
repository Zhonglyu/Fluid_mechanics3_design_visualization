import numpy as np


def concentration(x, y, z, t,
                  m_dot,      # 排放率 \dot m
                  Ux, Uy, Um, # 背景流 Ux, 横向风 Uy, 船速 Um
                  DT,         # 横向/竖向湍扩散系数 D_T
                  h):         # 源高度 h（到海面距离）

    """
    高斯羽流 + perfect reflector 的浓度场 C(x, y, z, t)

    参数支持 numpy 数组，所以可以直接传入 meshgrid.
    """

    U_rel = Ux + Um  # 相对流速

    # sigma^2 = 2 D_T (x - Um t) / (Ux + Um)
    sigma2 = 2.0 * DT * (x - Um * t) / U_rel

    # 为了避免上游 (x < Um t) 出现负的 sigma^2，直接把那一侧浓度设为 0
    sigma2 = np.where(sigma2 > 0, sigma2, np.nan)

    # 公共的横向位移项: y - Uy/(Ux+Um) * (x - Ux t)
    y0 = y - (Uy / U_rel) * (x - Ux * t)

    # 两个镜像源：z = h 和 z = -h
    denom = 2.0 * sigma2
    prefactor = m_dot / (2.0 * np.pi * U_rel * sigma2)

    expo1 = - (y0**2 + (z - h)**2) / denom
    expo2 = - (y0**2 + (z + h)**2) / denom

    C = prefactor * (np.exp(expo1) + np.exp(expo2))

    # 把 NaN 的地方（上游）改成 0，更方便画图
    C = np.nan_to_num(C, nan=0.0)

    return C
