import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from IPython.display import HTML
from GD_utils import make_2D_loss_surface

from typing import cast
from matplotlib.figure import Figure

def animate_GD(
    X,
    wi,
    bi,
    GDs, 
    labels=[], 
    skip_epochs=10, 
    save_as=None, 
    frames=200, 
    interval=50
  ):
  contour = make_2D_loss_surface(X, wi, bi)
  fig, ax = contour.figure, contour.axes
  fig = cast(Figure, fig)

  w, b, Z = getattr(contour, "_grid")
  contour = ax.contourf(w, b, Z, levels=20)
  ax.set_title(f"Epoch: 1", fontsize=14, color="black", pad=20)

  gd_info = []
  for i in range(len(GDs)):
    point, = ax.plot([], [], 'ro', zorder=3)
    path, = ax.plot([], [], zorder=2, label=labels[i])
    w_hist, b_hist = [], []
    gd_info.append((point, path, (w_hist, b_hist)))

  def update(frame, GDs):
    artists = []
    for i in range(len(GDs)):
      point, path, (w_hist, b_hist) = gd_info[i]

      w, b = next(GDs[i])
      w_hist.append(w)
      b_hist.append(b)

      point.set_data([w], [b])
      path.set_data(w_hist, b_hist)

      if frame == 0:
        ax.plot(w, b, 'o', zorder=3, color="black")

      artists.extend([point, path])

    ax.set_title(f"Epoch: {(frame + 1) * skip_epochs}", fontsize=14, color="black", pad=20)

    return artists

  ani = FuncAnimation(fig, update, fargs=(GDs,), frames=frames, interval=interval, blit=True, repeat=True)
  plt.close(fig)
  ax.legend(loc="upper right")
  ax.set_xlabel("w")
  ax.set_ylabel("b")

  if save_as:
    writer = FFMpegWriter(fps=15, bitrate=1800)
    return ani.save(f"{"".join(save_as.split(".")[:-1])}.mp4", writer=writer)
  return HTML(ani.to_jshtml())