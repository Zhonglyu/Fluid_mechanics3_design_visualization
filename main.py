import matplotlib.pyplot as plt
import numpy as np
from visualisation import concentration
import plotly.graph_objects as go



# 参数示例
M_DOT = 100.0      # g/s
Ux, Uy, Um = -0.5, 0.1, 1
DT = 0.2
H = 1.0
Z0 = -1.0         # 取和源高度差不多的一个水平面
t = 30.0



# 网格

X, Y, Z = np.mgrid[-5:500:50j, -5:100:50j, -50:0:50j]
# print(concentration(, Y[0], Z[0], t, M_DOT, Ux, Uy, Um, DT, H))

C = concentration(X, Y, Z, t, M_DOT, Ux, Uy, Um, DT, H)
print(f"浓度范围: min={C.min():.6f}, max={C.max():.6f}, 非零点数={np.count_nonzero(C)}")

values = C

fig = go.Figure(data=go.Volume(
    x=X.flatten(),
    y=Y.flatten(),
    z=Z.flatten(),
    value=values.flatten(),
    isomin=C.min(),
    isomax=5,
    opacity=0.1, # needs to be small to see through all surfaces
    surface_count=1000, # needs to be a large number for good volume rendering
    ))
fig.show()