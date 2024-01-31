import numpy as np
import matplotlib.pyplot as plt

n = 9
n_hr = 1001

# %%
x, dx = np.linspace(-4, 4, n, retstep=True)
y, dy = np.linspace(-4, 4, n, retstep=True)
X, Y = np.meshgrid(x, y, indexing="ij")

R = np.sqrt((X / 1.5) ** 2 + Y**2)
image = (np.cos(X) * np.sin(Y / 2)) * np.exp(-0.08 * (R**2))

# %%
x_hr, dx_hr = np.linspace(-4, 4, n_hr, retstep=True)
y_hr, dy_hr = np.linspace(-4, 4, n_hr, retstep=True)
X_hr, Y_hr = np.meshgrid(x_hr, y_hr, indexing="ij")

R_hr = np.sqrt((X_hr / 1.5) ** 2 + Y_hr**2)
image_hr = (np.cos(X_hr) * np.sin(Y_hr / 2)) * np.exp(-0.08 * (R_hr**2))


# %%

fig, ax = plt.subplots(figsize=(6, 6), tight_layout=True)
im = ax.imshow(
    image.T,
    cmap="Greys",
    origin="lower",
    extent=[
        x.min() - 0.5 * dx,
        x.max() + 0.5 * dx,
        y.min() - 0.5 * dy,
        y.max() + 0.5 * dy,
    ],
)

for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        ax.text(
            x[i],
            y[j] - 0.25 * dy,
            f"{image[i, j]:.2f}",
            color="r",
            ha="center",
            va="center",
        )
        ax.text(
            x[i],
            y[j] + 0.25 * dy,
            f"({i},{j})",
            color=plt.cm.tab10(1),
            ha="center",
            va="center",
        )

ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")

fig.colorbar(im, ax=ax, location="right", fraction=0.02)
fig.show()

# %%

fig2 = plt.figure(tight_layout=True, figsize=(6, 6))
ax2 = fig2.add_subplot(111, projection="3d")
ax2.plot_surface(X_hr, Y_hr, image_hr, cmap="Greys")
ax2.set_xlabel(r"$x$")
ax2.set_ylabel(r"$y$")
fig2.show()

# %%
fig3, ax3 = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True)
im30 = ax3[0].imshow(
    image_hr.T,
    cmap="Greys",
    origin="lower",
    extent=[
        x_hr.min() - 0.5 * dx_hr,
        x_hr.max() + 0.5 * dx_hr,
        y_hr.min() - 0.5 * dy_hr,
        y_hr.max() + 0.5 * dy_hr,
    ],
)
im31 = ax3[1].contour(X_hr, Y_hr, image_hr, levels=20, origin="lower")
ax3[1].clabel(im31, im31.levels, inline=True, fontsize=6)
ax3[1].set_aspect("equal")

for axx in ax3:
    axx.set_xlabel(r"$x$")
    axx.set_ylabel(r"$y$")

fig3.colorbar(im30, ax=ax3[0], location="right", fraction=0.02)
fig3.colorbar(im31, ax=ax3[1], location="right", fraction=0.02)
fig3.show()

# %%

fig4 = plt.figure(tight_layout=True, figsize=(6, 6))
ax4 = fig4.add_subplot(111, projection="3d")
ax4.plot_surface(X, Y, image, cmap="Greys")
ax4.set_xlabel(r"$x$")
ax4.set_ylabel(r"$y$")
fig4.show()
