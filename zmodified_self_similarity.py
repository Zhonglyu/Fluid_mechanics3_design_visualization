import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from visualisation import concentration

# 主题
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

# ========= 物理参数（保留初始化，后续逐张图按需补充） =========
m_dot = 1.0
Ux, Uy, Um = -0.5, 0.1, 1.0
DT = 0.3
h = 1.0
t = 50.0

U_rel = -Ux + Um
x_sections = [-10.0, -30.0, -50.0, -70.0, -100.0]

# 采样网格
y_rel = np.linspace(-30.0, 30.0, 400)
z_rel = np.linspace(-30.0, 30.0, 400)


def plot_cy_vs_y():
    """第一张图：横轴 y_rel，纵轴 C_y = C(x, y, z=h)。"""
    rows = []
    for x in x_sections:
        x_rel = x - Um * t
        sigma2 = 2.0 * DT * x_rel / U_rel
        if (sigma2 >= 0) or (t <= 0):
            continue
        y_center = -(Uy / U_rel) * x_rel
        y_phys = y_center + y_rel
        C_y = concentration(x, y_phys, h, t, m_dot, Ux, Uy, Um, DT, h)
        label = f"x = {int(abs(x))} m"
        rows.append(pd.DataFrame({"y_rel": y_rel, "C_y": C_y, "section": label}))

    df = pd.concat(rows, ignore_index=True)
    palette = sns.cubehelix_palette(len(df["section"].unique()), rot=-0.25, light=0.7)

    g = sns.FacetGrid(
        df,
        row="section",
        hue="section",
        aspect=8,
        height=0.7,
        palette=palette,
        sharex=True,
        sharey=False,
    )

    g.map(
        sns.kdeplot,
        "y_rel",
        bw_adjust=0.4,
        clip_on=False,
        fill=True,
        alpha=1,
        linewidth=1.2,
        weights=df["C_y"],
    )
    g.map(
        sns.kdeplot,
        "y_rel",
        clip_on=False,
        color="w",
        lw=1.4,
        bw_adjust=0.4,
        weights=df["C_y"],
    )

    g.refline(y=0, linewidth=1, linestyle="-", color=None, clip_on=False)

    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, 0.2, label, fontweight="bold", color=color, ha="left", va="center", transform=ax.transAxes)

    g.map(label, "y_rel")
    # 调整布局与轴显示
    g.figure.subplots_adjust(hspace=0.05)
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.set_xlabels("y [m]")
    # 保留 x 轴刻度
    for ax in g.axes.flatten():
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
    g.despine(bottom=False, left=False)
    g.fig.text(0.02, 0.5, r"C(x,y,0)", va="center", rotation="vertical", fontsize=10)
    g.fig.suptitle(f"C vs y at different x at t = {t}s", y=1.02, fontsize=12, fontweight="bold")


def plot_cy_self_vs_eta_y():
    """第二张图：横轴 eta_y，纵轴 C_y/C_center。"""
    rows = []
    for x in x_sections:
        x_rel = x - Um * t
        sigma2 = 2.0 * DT * x_rel / U_rel
        if (sigma2 >= 0) or (t <= 0):
            continue
        sigma = np.sqrt(-sigma2)
        y_center = -(Uy / U_rel) * x_rel
        y_phys = y_center + y_rel
        C_y = concentration(x, y_phys, h, t, m_dot, Ux, Uy, Um, DT, h)
        C_center = concentration(x, y_center, h, t, m_dot, Ux, Uy, Um, DT, h)
        eta_y = y_rel / sigma
        C_y_self = C_y / C_center
        label = f"x = {int(abs(x))} m"
        rows.append(pd.DataFrame({"eta_y": eta_y, "C_ratio": C_y_self, "section": label}))

    df = pd.concat(rows, ignore_index=True)
    palette = sns.cubehelix_palette(len(df["section"].unique()), rot=-0.25, light=0.7)

    g = sns.FacetGrid(
        df,
        row="section",
        hue="section",
        aspect=8,
        height=0.7,
        palette=palette,
        sharex=True,
        sharey=False,
    )

    g.map(
        sns.kdeplot,
        "eta_y",
        bw_adjust=0.4,
        clip_on=False,
        fill=True,
        alpha=1,
        linewidth=1.2,
        weights=df["C_ratio"],
    )
    g.map(
        sns.kdeplot,
        "eta_y",
        clip_on=False,
        color="w",
        lw=1.4,
        bw_adjust=0.4,
        weights=df["C_ratio"],
    )

    g.refline(y=0, linewidth=1, linestyle="-", color=None, clip_on=False)

    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, 0.2, label, fontweight="bold", color=color, ha="left", va="center", transform=ax.transAxes)

    g.map(label, "eta_y")
    g.figure.subplots_adjust(hspace=0.05)
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.set_xlabels(r"$\eta_y = y / \sigma_y(x)$")
    for ax in g.axes.flatten():
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
    g.despine(bottom=False, left=False)
    g.fig.text(0.02, 0.5, r"$C / C_c(x)$", va="center", rotation="vertical", fontsize=10)
    g.fig.suptitle(f"Horizontal self similarity plot at t = {t}s", y=1.02, fontsize=12, fontweight="bold")


def plot_cz_vs_z():
    """第三张图：横轴 z_rel，纵轴 C_z = C(x, y_center, z)。"""
    rows = []
    for x in x_sections:
        x_rel = x - Um * t
        sigma2 = 2.0 * DT * x_rel / U_rel
        if (sigma2 >= 0) or (t <= 0):
            continue
        y_center = -(Uy / U_rel) * x_rel
        z_phys = h + z_rel
        C_z = concentration(x, y_center, z_phys, t, m_dot, Ux, Uy, Um, DT, h)
        label = f"x = {int(abs(x))} m"
        rows.append(pd.DataFrame({"z_rel": z_rel, "C_z": C_z, "section": label}))

    df = pd.concat(rows, ignore_index=True)
    palette = sns.cubehelix_palette(len(df["section"].unique()), rot=-0.25, light=0.7)

    g = sns.FacetGrid(
        df,
        row="section",
        hue="section",
        aspect=8,
        height=0.7,
        palette=palette,
        sharex=True,
        sharey=False,
    )

    g.map(
        sns.kdeplot,
        "z_rel",
        bw_adjust=0.4,
        clip_on=False,
        fill=True,
        alpha=1,
        linewidth=1.2,
        weights=df["C_z"],
    )
    g.map(
        sns.kdeplot,
        "z_rel",
        clip_on=False,
        color="w",
        lw=1.4,
        bw_adjust=0.4,
        weights=df["C_z"],
    )

    g.refline(y=0, linewidth=1, linestyle="-", color=None, clip_on=False)

    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, 0.2, label, fontweight="bold", color=color, ha="left", va="center", transform=ax.transAxes)

    g.map(label, "z_rel")
    g.figure.subplots_adjust(hspace=0.05)
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.set_xlabels("z [m]")
    for ax in g.axes.flatten():
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
    g.despine(bottom=False, left=False)
    g.fig.text(0.02, 0.5, r"C(x,0,z)", va="center", rotation="vertical", fontsize=10)
    g.fig.suptitle(f"C vs z at different x at t = {t}s", y=1.02, fontsize=12, fontweight="bold")


def plot_cz_self_vs_eta_z():
    """第四张图：横轴 eta_z = z_rel/σ，纵轴 C_z/C_center。"""
    rows = []
    for x in x_sections:
        x_rel = x - Um * t
        sigma2 = 2.0 * DT * x_rel / U_rel
        if (sigma2 >= 0) or (t <= 0):
            continue
        sigma = np.sqrt(-sigma2)
        y_center = -(Uy / U_rel) * x_rel
        z_phys = h + z_rel
        C_z = concentration(x, y_center, z_phys, t, m_dot, Ux, Uy, Um, DT, h)
        C_center = concentration(x, y_center, h, t, m_dot, Ux, Uy, Um, DT, h)
        eta_z = z_rel / sigma
        C_z_self = C_z / C_center
        label = f"x = {int(abs(x))} m"
        rows.append(pd.DataFrame({"eta_z": eta_z, "C_ratio": C_z_self, "section": label}))

    df = pd.concat(rows, ignore_index=True)
    palette = sns.cubehelix_palette(len(df["section"].unique()), rot=-0.25, light=0.7)

    g = sns.FacetGrid(
        df,
        row="section",
        hue="section",
        aspect=8,
        height=0.7,
        palette=palette,
        sharex=True,
        sharey=False,
    )

    g.map(
        sns.kdeplot,
        "eta_z",
        bw_adjust=0.4,
        clip_on=False,
        fill=True,
        alpha=1,
        linewidth=1.2,
        weights=df["C_ratio"],
    )
    g.map(
        sns.kdeplot,
        "eta_z",
        clip_on=False,
        color="w",
        lw=1.4,
        bw_adjust=0.4,
        weights=df["C_ratio"],
    )

    g.refline(y=0, linewidth=1, linestyle="-", color=None, clip_on=False)

    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, 0.2, label, fontweight="bold", color=color, ha="left", va="center", transform=ax.transAxes)

    g.map(label, "eta_z")
    g.figure.subplots_adjust(hspace=0.05)
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.set_xlabels(r"$\eta_z = z / \sigma_z(x)$")
    for ax in g.axes.flatten():
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
    g.despine(bottom=False, left=False)
    g.fig.text(0.02, 0.5, r"$C / C_c(x)$", va="center", rotation="vertical", fontsize=10)
    g.fig.suptitle(f"Vertical self similarity at t = {t}s", y=1.02, fontsize=12, fontweight="bold")


def plot_all_in_grid():
    """将四张图按 2x2 排列在同一画布上（线图简化版，保持相同变量与标注）。"""
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    cmap = sns.cubehelix_palette(len(x_sections), rot=-0.25, light=0.7)

    def compute_profiles():
        profiles = []
        for x in x_sections:
            x_rel = x - Um * t
            sigma2 = 2.0 * DT * x_rel / U_rel
            if (sigma2 >= 0) or (t <= 0):
                continue
            sigma = np.sqrt(-sigma2)
            y_center = -(Uy / U_rel) * x_rel
            y_phys = y_center + y_rel
            z_phys = h + z_rel
            C_y = concentration(x, y_phys, h, t, m_dot, Ux, Uy, Um, DT, h)
            C_center = concentration(x, y_center, h, t, m_dot, Ux, Uy, Um, DT, h)
            C_y_self = C_y / C_center
            C_z = concentration(x, y_center, z_phys, t, m_dot, Ux, Uy, Um, DT, h)
            C_z_self = C_z / C_center
            eta_y = y_rel / sigma
            eta_z = z_rel / sigma
            profiles.append((x, C_y, C_y_self, C_z, C_z_self, eta_y, eta_z))
        return profiles

    profiles = compute_profiles()

    # 1) C vs y
    ax = axs[0, 0]
    for (x, C_y, _, _, _, _, _) in profiles:
        label = f"x = {int(abs(x))} m"
        ax.plot(y_rel, C_y, label=label)
    ax.set_xlabel("y [m]")
    ax.set_ylabel(r"C(x,y,0)")
    ax.set_title(f"C vs y at different x at t = {t}s")
    ax.grid(True, linestyle="--", alpha=0.3)

    # 2) C/C_center vs eta_y
    ax = axs[0, 1]
    for (x, _, C_y_self, _, _, eta_y, _) in profiles:
        label = f"x = {int(abs(x))} m"
        ax.plot(eta_y, C_y_self, label=label)
    ax.set_xlabel(r"$\eta_y = y / \sigma_y(x)$")
    ax.set_ylabel(r"$C / C_c(x)$")
    ax.set_title(f"Horizontal self similarity plot at t = {t}s")
    ax.grid(True, linestyle="--", alpha=0.3)

    # 3) C vs z
    ax = axs[1, 0]
    for (x, _, _, C_z, _, _, _) in profiles:
        label = f"x = {int(abs(x))} m"
        ax.plot(z_rel, C_z, label=label)
    ax.set_xlabel("z [m]")
    ax.set_ylabel(r"C(x,0,z)")
    ax.set_title(f"C vs z at different x at t = {t}s")
    ax.grid(True, linestyle="--", alpha=0.3)

    # 4) C/C_center vs eta_z
    ax = axs[1, 1]
    for (x, _, _, _, C_z_self, _, eta_z) in profiles:
        label = f"x = {int(abs(x))} m"
        ax.plot(eta_z, C_z_self, label=label)
    ax.set_xlabel(r"$\eta_z = z / \sigma_z(x)$")
    ax.set_ylabel(r"$C / C_c(x)$")
    ax.set_title(f"Vertical self similarity at t = {t}s")
    ax.grid(True, linestyle="--", alpha=0.3)

    # 统一图例放在外侧
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(labels),
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, 0.99),
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])


def plot_all_ridge_in_grid():
    """四张 ridge plot（KDE+offset）放在同一 2x2 画布。"""
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    palette = sns.cubehelix_palette(len(x_sections), rot=-0.25, light=0.7)

    profiles = []
    for x in x_sections:
        x_rel = x - Um * t
        sigma2 = 2.0 * DT * x_rel / U_rel
        if (sigma2 >= 0) or (t <= 0):
            continue
        sigma = np.sqrt(-sigma2)
        y_center = -(Uy / U_rel) * x_rel
        y_phys = y_center + y_rel
        z_phys = h + z_rel
        C_y = concentration(x, y_phys, h, t, m_dot, Ux, Uy, Um, DT, h)
        C_center = concentration(x, y_center, h, t, m_dot, Ux, Uy, Um, DT, h)
        C_y_self = C_y / C_center
        C_z = concentration(x, y_center, z_phys, t, m_dot, Ux, Uy, Um, DT, h)
        C_z_self = C_z / C_center
        eta_y = y_rel / sigma
        eta_z = z_rel / sigma
        profiles.append(
            dict(
                label=f"x = {int(abs(x))} m",
                y_rel=y_rel,
                C_y=C_y,
                eta_y=eta_y,
                C_y_self=C_y_self,
                z_rel=z_rel,
                C_z=C_z,
                eta_z=eta_z,
                C_z_self=C_z_self,
            )
        )

    def ridge_axis(ax, coord_key, value_key, xlabel, ylabel, title):
        offset_step = 1.2
        for i, prof in enumerate(profiles):
            coord = prof[coord_key]
            values = prof[value_key]
            kde = gaussian_kde(coord, weights=values, bw_method=0.4)
            grid = np.linspace(coord.min(), coord.max(), 400)
            dens = kde(grid)
            dens = dens / (dens.max() + 1e-9)
            base = i * offset_step
            color = palette[i % len(palette)]
            ax.fill_between(grid, base, base + dens, color=color, alpha=0.7)
            ax.plot(grid, base + dens, color="white", linewidth=1.0)
            ax.text(grid.min(), base + 0.05, prof["label"], color=color, fontsize=8, va="bottom")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.3)

    ridge_axis(
        axs[0, 0],
        "y_rel",
        "C_y",
        "y [m]",
        r"C(x,y,0)",
        f"C vs y at different x at t = {t}s",
    )
    ridge_axis(
        axs[0, 1],
        "eta_y",
        "C_y_self",
        r"$\eta_y = y / \sigma_y(x)$",
        r"$C / C_c(x)$",
        f"Horizontal self similarity plot at t = {t}s",
    )
    ridge_axis(
        axs[1, 0],
        "z_rel",
        "C_z",
        "z [m]",
        r"C(x,0,z)",
        f"C vs z at different x at t = {t}s",
    )
    ridge_axis(
        axs[1, 1],
        "eta_z",
        "C_z_self",
        r"$\eta_z = z / \sigma_z(x)$",
        r"$C / C_c(x)$",
        f"Vertical self similarity at t = {t}s",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.suptitle(f"Ridge plots at t = {t}s", fontsize=14, fontweight="bold")


if __name__ == "__main__":
    # plot_cy_vs_y()
    # plot_cy_self_vs_eta_y()
    # plot_cz_vs_z()
    # plot_cz_self_vs_eta_z()
    plot_all_in_grid()  # 保留线图版本（需要时可取消注释）
    plot_all_ridge_in_grid()  # 四张 ridge plot 汇总在一张大图
    plt.show()

