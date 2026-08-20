import numpy as np
import matplotlib.pyplot as plt

from mdm_hmf import (suppression_ratio, model_parameters)


# -------------
# Example model
# -------------

f = 0.1
z = 2.0

M = np.logspace(np.log10(2e10), 13, 300)


# ---------------------
# Calculate suppression
# ---------------------

R = suppression_ratio(M, f=f, z=z)


M_S, alpha, beta = model_parameters(f=f, z=z)

print(f"f       = {f}")
print(f"z       = {z}")
print(f"M_S     = {M_S:.3e} Msun/h")
print(f"alpha   = {alpha:.4f}")
print(f"beta    = {beta:.4f}")


# ----
# Plot
# ----

fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(M, R, lw=2)

ax.set_xscale("log")
ax.set_xlabel(r"$M_{200c}\,[h^{-1}\,M_\odot]$")
ax.set_ylabel(r"$R = "
    r"(\mathrm{d}n/\mathrm{d}\log_{10}M)_{\rm MDM}"
    r"/"
    r"(\mathrm{d}n/\mathrm{d}\log_{10}M)_{\rm CDM}$")
ax.set_title(rf"$f={f},\ z={z}$")

ax.axhline(1.0, linestyle="--", linewidth=1)

fig.tight_layout()

plt.savefig("suppression_example.pdf", bbox_inches="tight")

plt.show()
