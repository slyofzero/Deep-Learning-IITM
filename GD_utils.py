import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# ============================== Sigmoid Function ==============================
def sigmoid(x, w, b):
  return 1 / (1 + np.exp(-(w*x + b)))

# ============================== Mean Squared Error ==============================
def MSE(X, Y, w, b):
  total_error = 0

  for x, y in zip(X, Y):
    y_pred = sigmoid(x, w, b)
    total_error += np.square(y - y_pred)

  return total_error / (2 * len(X))

def MSE_grad_w(y_pred: float, x: float, y: float):
  return (y_pred - y) * y_pred * (1 - y_pred) * x

def MSE_grad_b(y_pred: float, y: float):
  return (y_pred - y) * y_pred * (1 - y_pred)

# ============================== Loss surface ==============================
def prepare_loss_surface(X, wi, bi, function="sigmoid", loss="MSE", mu=10):
  '''
  `X` is the input.\n
  `wi` is the original function's weight.\n
  `bi` is the original function's bias.\n
  `mu` is to create a range around both `w` and `b`.
  '''

  g = sigmoid if function == "sigmoid" else sigmoid
  loss = MSE if function == "MSE" else MSE

  X = np.array([-2, 0.15, -1.1, 0, -0.1, 1])
  Y = g(X, wi, bi)

  w = np.linspace(wi-mu, wi+mu, 100)
  b = np.linspace(bi-mu, bi+mu, 100)
  W, B = np.meshgrid(w, b)
  Z = loss(X, Y, W, B)

  return w, b, W, B, Z

# ========== Filled Contour Plot ==========
def make_2D_loss_surface(X, wi, bi, function="sigmoid", loss="MSE", mu=10):
  w, b, *_, Z = prepare_loss_surface(X, wi, bi, function, loss, mu)

  fig, ax = plt.subplots(figsize=(10, 8))
  contour = ax.contourf(w, b, Z, levels=100, cmap="viridis")
  setattr(contour, "_grid", (w, b, Z))
  fig.colorbar(contour, ax=ax, label="Error")
  ax.set_xlabel("w")
  ax.set_ylabel("b")
  ax.set_title("Error Surface Contour Plot")

  return contour

# ========== Plotly 3D surface Plot ==========
def make_3D_loss_surface(X, wi, bi, function="sigmoid", loss="MSE", mu=10):
  *_, W, B, Z = prepare_loss_surface(X, wi, bi, function, loss, mu)

  fig = go.Figure(data=[go.Surface(x=W, y=B, z=Z, colorscale="Viridis")])
  fig.update_layout(
      scene=dict(
          xaxis_title="w",
          yaxis_title="b",
          zaxis_title="loss"
      ),
      width=1000,
      height=800,
      title="Error Surface (3D)"
  )
  fig.show()
  
  return fig