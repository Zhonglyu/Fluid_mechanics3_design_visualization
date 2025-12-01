import numpy as np
from dash import Dash, Input, Output, dcc, html
import plotly.graph_objects as go
from visualisation import concentration


# 参数示例
M_DOT = 100.0      # g/s
Ux, Uy, Um = -0.5, 0.1, 1
DT = 0.2
H = 1.0
Z0 = -1.0         # 取和源高度差不多的一个水平面
t = 30.0


X, Y, Z = np.mgrid[-5:500:100j, -5:100:100j, -50:0:100j]
X_FLAT = X.flatten()
Y_FLAT = Y.flatten()
Z_FLAT = Z.flatten()


def build_volume_figure(current_t: float) -> go.Figure:
    """根据时间 current_t 计算浓度场并生成体渲染图。"""
    C = concentration(X, Y, Z, current_t, M_DOT, Ux, Uy, Um, DT, H)
    c_min = float(np.min(C))
    c_max = float(np.max(C))
    if c_max <= 0:
        c_max = 1.0
    fig = go.Figure(data=go.Volume(
        x=X_FLAT,
        y=Y_FLAT,
        z=Z_FLAT,
        value=C.flatten(),
        isomin=c_min,
        isomax=c_max,
        opacity=0.1,  # needs to be small to see through all surfaces
        surface_count=60,  # 平衡渲染效果和 Dash 回调性能
    ))
    fig.update_layout(
        title=f"t = {current_t:.1f} s 的浓度场 (C)",
        scene=dict(
            xaxis_title="x (m)",
            yaxis_title="y (m)",
            zaxis_title="z (m)",
        ),
    )
    return fig



app = Dash(__name__)
app.layout = html.Div(
    [
        html.H3("羽流浓度场 (C) 与时间滑动器"),
        dcc.Slider(
            id="time-slider",
            min=0,
            max=120,
            step=1,
            value=t,
            marks={0: "0s", 30: "30s", 60: "60s", 90: "90s", 120: "120s"},
            tooltip={"placement": "bottom", "always_visible": True},
        ),
        html.Div(id="time-readout", style={"margin": "12px 0"}),
        dcc.Graph(
            id="concentration-volume",
            figure=build_volume_figure(t),
            style={"height": "80vh"},
        ),
    ]
)


@app.callback(
    Output("concentration-volume", "figure"),
    Output("time-readout", "children"),
    Input("time-slider", "value"),
)
def update_volume(time_value: float):
    C = concentration(X, Y, Z, time_value, M_DOT, Ux, Uy, Um, DT, H)
    c_min = float(np.min(C))
    c_max = float(np.max(C))
    readout = (
        f"当前 t = {time_value:.1f} s | C_min = {c_min:.4f}, "
        f"C_max = {c_max:.4f}"
    )
    fig = go.Figure(data=go.Volume(
        x=X_FLAT,
        y=Y_FLAT,
        z=Z_FLAT,
        value=C.flatten(),
        isomin=c_min,
        isomax=c_max if c_max > 0 else 1.0,
        opacity=0.1,
        surface_count=60,
    ))
    fig.update_layout(
        title=f"t = {time_value:.1f} s 的浓度场 (C)",
        scene=dict(
            xaxis_title="x (m)",
            yaxis_title="y (m)",
            zaxis_title="z (m)",
        ),
    )
    return fig, readout


if __name__ == "__main__":
    app.run(debug=True)