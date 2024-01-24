# %%
print('a')
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.size"] = 8

# %%
n = 16
x = np.arange(n)
k = np.arange(n // 2 + 1)

f = np.zeros(n)
f[1] = 2
f[2:7] = 4
f[7] = 2
f[8:11] = 1
f[14] = 8

# f = 2.5 * np.cos(6 * (2 * np.pi) * x / n) + 1
# f -= f.mean()
# f = np.exp(-((x - 9) ** 2) / (2 * 2**2))

# %%
F = np.fft.rfft(f, norm="forward")
F_amp = np.abs(F)
F_amp[1:-1] *= 2
F_phase = np.angle(F)
F_phase[F_amp < 1e-5] = 0

F_phase[0] = 0
F_phase[-1] = 0

# %%
fig, ax = plt.subplots(3, 1, figsize=(4, 5))
ax[0].bar(x, f, width=(x[1] - x[0]), edgecolor="k")
ax[1].bar(k, F_amp, width=(k[1] - k[0]), edgecolor="k", color=plt.cm.tab10(1))
ax[2].bar(k, F_phase, width=(k[1] - k[0]), edgecolor="k", color=plt.cm.tab10(1))
ax[2].set_ylim(-np.pi, np.pi)

ax[0].set_title("signal $f(x)$")
ax[1].set_title("cosine amplitudes")
ax[2].set_title("cosine phases")

ax[0].set_xlabel("location $x$")
ax[1].set_xlabel("spatial frequency $k$")
ax[2].set_xlabel("spatial frequency $k$")

fig.tight_layout()
fig.show()

# %%
fig2, ax2 = plt.subplots(n // 2 + 1, 3, figsize=(11, 8), sharex=True, sharey="col")

f_rec = np.zeros(n)

x2 = np.linspace(0, x[-1], 1000)

for i in k:
    tmp = F_amp[i] * np.cos(i * (2 * np.pi) * x / n + F_phase[i])
    f_rec += tmp

    tmp2 = F_amp[i] * np.cos(i * (2 * np.pi) * x2 / n + F_phase[i])

    ax2[i, 0].bar(
        x, np.cos(i * (2 * np.pi) * x / n), width=(x[1] - x[0]), edgecolor="k"
    )
    ax2[i, 0].plot(
        x2, np.cos(i * (2 * np.pi) * x2 / n), "-", lw=1, color=plt.cm.tab10(3)
    )

    ax2[i, 1].bar(x, tmp, width=(x[1] - x[0]), edgecolor="k")
    ax2[i, 1].plot(x2, tmp2, "-", lw=1, color=plt.cm.tab10(3))

    ax2[i, 2].bar(x, f_rec, width=(x[1] - x[0]), edgecolor="k")

    ax2[i, 0].set_title(
        f"basis function {i}: $\cos({i} \cdot 2\pi x / {n})$", fontsize="medium"
    )

    if i == 0 or i == n // 2:
        ax2[i, 1].set_title(
            f"${F_amp[i]:.2f} \cdot \cos({i} \cdot 2\pi x / {n})$",
            fontsize="medium",
        )
    else:
        ax2[i, 1].set_title(
            f"${F_amp[i]:.2f} \cdot \cos({i} \cdot 2\pi x / {n} {F_phase[i]:+.2f})$",
            fontsize="medium",
        )

    ax2[i, 2].set_title(f"sum of first ${i+1}$ cosines", fontsize="medium")

fig2.tight_layout()
fig2.show()
