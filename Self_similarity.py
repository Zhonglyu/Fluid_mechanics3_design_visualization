import numpy as np
import matplotlib.pyplot as plt
from visualisation import concentration


# ====== 参数（你可以按自己情况改） ======
m_dot = 1.0
Ux, Uy, Um = -0.5, 0.1, 1.0
DT = 0.3
h = 1.0
t = 50.0       # 固定一个观察时间

U_rel = - Ux

# 在哪些下游 x 截面上画剖面（这里是“你传进函数的 x”）
x_sections = [-20.0, -50.0, -100.0]

# 相对坐标：相对于中心轴的 y', z'
y_rel = np.linspace(-30, 30, 400)
z_rel = np.linspace(-30, 30, 400)

# ====== 开始画图 ======
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
ax_y_raw, ax_y_self = axes[0]
ax_z_raw, ax_z_self = axes[1]

for x in x_sections:
    # 和你的公式保持一致：先算 x_rel 和 sigma
    x_rel = x - Um * t
    sigma2 = 2.0 * DT * x_rel / U_rel

    # 跳过无效的截面
    if (sigma2 >= 0) or (t <= 0):
        continue

    sigma = np.sqrt(-sigma2)

    # ==============================
    # 1. 横向剖面 C(x, y, z=h)
    #    先算出中心轴在实验室坐标系里的 y_center
    # ==============================
    y_center = -(Uy / U_rel) * x_rel   # 让 y_rel = 0 的位置落在浓度中心

    # 实际代入函数的 y 坐标：y = y_center + y_rel
    y_phys = y_center + y_rel

    # 在 z = h 的平面上看横向剖面
    C_y = concentration(x, y_phys, h, t, m_dot, Ux, Uy, Um, DT, h)

    # 中心轴浓度（y_rel=0, z=h）
    C_center = concentration(x, y_center, h, t, m_dot, Ux, Uy, Um, DT, h)

    # 自相似变量
    eta_y = y_rel / sigma

    # 原始剖面（横坐标用相对 y'，中心在 0）
    ax_y_raw.plot(y_rel, C_y, label=f"x = {x:g}")

    # self-similarity：用 C/C_center 对 eta_y
    ax_y_self.plot(eta_y, C_y / C_center, label=f"x = {x:g}")

    # ==============================
    # 2. 竖向剖面 C(x, y_center, z)
    #    固定 y 在中心轴 y_center，扫描 z
    # ==============================
    z_phys = h + z_rel   # 以 z = h 为中心画剖面
    C_z = concentration(x, y_center, z_phys, t, m_dot, Ux, Uy, Um, DT, h)

    # 竖向也用同一个 sigma（因为你现在 DT 是各向同性）
    eta_z = z_rel / sigma

    ax_z_raw.plot(z_rel, C_z, label=f"x = {x:g}")
    ax_z_self.plot(eta_z, C_z / C_center, label=f"x = {x:g}")


# ====== 美化一下图像 ======
ax_y_raw.set_xlabel("y [m]")
ax_y_raw.set_ylabel("C(x, y, 0)")
ax_y_raw.set_title("C vs y at different x at t = {}s".format(t))

ax_y_self.set_xlabel(r"$\eta_y = y / \sigma_y(x)$")
ax_y_self.set_ylabel(r"$C / C_c(x)$")
ax_y_self.set_title("Horitontal self similarty plot at t = {}s".format(t))

ax_z_raw.set_xlabel("z [m]")
ax_z_raw.set_ylabel("C(x, 0, z)")
ax_z_raw.set_title("C vs z at different x at t = {}s".format(t))

ax_z_self.set_xlabel(r"$\eta_z = z / \sigma_z(x)$")
ax_z_self.set_ylabel(r"$C / C_c(x)$")
ax_z_self.set_title("Vertical Self similarity at t = {}s".format(t))

for ax in [ax_y_raw, ax_y_self, ax_z_raw, ax_z_self]:
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

plt.tight_layout()
plt.show()