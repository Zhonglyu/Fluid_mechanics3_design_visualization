import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# 参数区：按你自己的工况改
# ----------------------------
M_dot = 1.0        # 连续释放率 [kg/s]，只影响绝对数值，不影响归一化形状
U = 0.5            # 主流速度 [m/s]
D_y = 0.1          # 横向湍扩散系数 [m^2/s]
D_z = 0.05         # 竖向湍扩散系数 [m^2/s]

# 在哪些下游位置看剖面（单位：m）
x_sections = [20.0, 50.0, 100.0]

# y、z 方向取值范围
y = np.linspace(-30, 30, 400)
z = np.linspace(-30, 30, 400)


# ----------------------------
# 模型函数
# ----------------------------
def sigma(D, x, U):
    """
    self-similar 宽度 sigma(x) = sqrt(2 D x / U)
    """
    x = np.asarray(x)
    return np.sqrt(2.0 * D * x / U)


def C_plume(x, y, z, M_dot, U, D_y, D_z):
    """
    高斯稳态连续羽流解：
    C(x,y,z) = M_dot / (2π U σ_y σ_z) * exp(- y^2/(2 σ_y^2) - z^2/(2 σ_z^2))
    """
    sig_y = sigma(D_y, x, U)
    sig_z = sigma(D_z, x, U)

    prefactor = M_dot / (2.0 * np.pi * U * sig_y * sig_z)

    # y,z 可能是数组，需要广播
    y = np.asarray(y)
    z = np.asarray(z)

    return prefactor * np.exp(-0.5 * ((y / sig_y) ** 2 + (z / sig_z) ** 2))


# ----------------------------
# 画图：C vs y / z & 自相似剖面
# ----------------------------
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
ax_y_raw, ax_y_self = axes[0]
ax_z_raw, ax_z_self = axes[1]

for x in x_sections:
    # ----- 横向剖面 C(x, y, 0) -----
    C_y = C_plume(x, y, 0.0, M_dot, U, D_y, D_z)
    sig_y = sigma(D_y, x, U)
    eta_y = y / sig_y
    Cc_x = C_plume(x, 0.0, 0.0, M_dot, U, D_y, D_z)  # 中心轴浓度

    ax_y_raw.plot(y, C_y, label=f"x = {x:g} m")
    ax_y_self.plot(eta_y, C_y / Cc_x, label=f"x = {x:g} m")

    # ----- 竖向剖面 C(x, 0, z) -----
    C_z = C_plume(x, 0.0, z, M_dot, U, D_y, D_z)
    sig_z = sigma(D_z, x, U)
    eta_z = z / sig_z

    ax_z_raw.plot(z, C_z, label=f"x = {x:g} m")
    ax_z_self.plot(eta_z, C_z / Cc_x, label=f"x = {x:g} m")


# ----------------------------
# 轴标签 & 图例
# ----------------------------
ax_y_raw.set_xlabel("y [m]")
ax_y_raw.set_ylabel("C(x, y, 0)")
ax_y_raw.set_title("C vs y at different x")

ax_y_self.set_xlabel(r"$\eta_y = y / \sigma_y(x)$")
ax_y_self.set_ylabel(r"$C / C_c(x)$")
ax_y_self.set_title("Horitontal self similarty plot")

ax_z_raw.set_xlabel("z [m]")
ax_z_raw.set_ylabel("C(x, 0, z)")
ax_z_raw.set_title("C vs z at different x")

ax_z_self.set_xlabel(r"$\eta_z = z / \sigma_z(x)$")
ax_z_self.set_ylabel(r"$C / C_c(x)$")
ax_z_self.set_title("Vertical Self similarity")

for ax in [ax_y_raw, ax_y_self, ax_z_raw, ax_z_self]:
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

plt.tight_layout()
plt.show()