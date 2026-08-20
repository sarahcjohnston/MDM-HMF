"""
mdm_hmf.py

Lightweight implementation of the calibrated mixed cold dark matter (CDM) + fuzzy dark matter (FDM) halo mass function suppression model.

The model gives
    R(M, f, z) =
        (dn/dlog10M)_MDM / (dn/dlog10M)_CDM

Halo masses are M_200c in units of Msun / h.

Reference:
    S. C. Johnston et al, Halo mass functions in mixed cold and fuzzy dark matter models, https://arxiv.org/abs/2606.06599
"""

import warnings
import numpy as np


# ------------------
# FITTED PARAMETERS
# ------------------
# Parameterisation:
#
# log10(M_S / (Msun/h)) = m0 + mf*f + mz*z
# alpha                  = a0 + af*f + az*z
# beta                   = b0 + bf*f + bz*z
#

FULL_FIT = {
    "m0":  15.2596545,
    "mf":  -1.83977331,
    "mz":  -0.127441298,

    "a0":  -0.566999637,
    "af":  20.6129592,
    "az":   1.72436382,

    "b0":  -0.0136099799,
    "bf":   0.209918254,
    "bz":   0.00109262303,
}


# ------------------
# CALIBRATED DOMAIN
#-------------------

#Minimum and maximum FDM fraction
# The model was calibrated using simulations at f = 0.01, 0.1, 0.2, and 0.3
f_min = 0.01
f_max = 0.30

#Minimum and maximum redshift
z_min = 1.0
z_max = 4.0

# Adopted lower resolution cut and upper cut.
m_min = 2.0e10  # Msun / h
m_max = 1.0e13  # Msun / h


# -----------------
# DEBUGGING CHECKS
# -----------------

def _check_domain(M, f, z):
    """
    Warn when the requested values extend outside the
    calibration domain.
    """

    M = np.asarray(M, dtype=float)
    f = np.asarray(f, dtype=float)
    z = np.asarray(z, dtype=float)

    if np.any(M <= 0):
        raise ValueError("Halo masses must be positive.")

    if np.any((f < f_min) | (f > f_max)):
        warnings.warn(
            f"FDM fraction outside the calibrated range "
            f"{f_min} <= f <= {f_max}.",
            RuntimeWarning,
            stacklevel=2,
        )

    if np.any((z < z_min) | (z > z_max)):
        warnings.warn(
            f"Redshift outside the calibrated range "
            f"{z_min} <= z <= {z_max}.",
            RuntimeWarning,
            stacklevel=2,
        )

    if np.any(M < m_min):
        warnings.warn(
            f"Masses below the adopted resolution cut "
            f"M_200c = {m_min:.2e} Msun/h were not used "
            f"to calibrate the model.",
            RuntimeWarning,
            stacklevel=2,
        )

    if np.any(M > m_max):
        warnings.warn(
            "Requested masses extend beyond the approximate "
            "mass range sampled by the simulations. "
            "The high-mass calibration limit varies with "
            "FDM fraction and redshift.",
            RuntimeWarning,
            stacklevel=2,
        )


def _get_parameters(parameters):
    """
    Return the nine fitted coefficients.
    """

    required = (
        "m0", "mf", "mz",
        "a0", "af", "az",
        "b0", "bf", "bz",
    )

    missing = [
        name for name in required
        if name not in parameters
    ]

    if missing:
        raise KeyError(
            "Missing fitted parameters: "
            + ", ".join(missing)
        )

    return tuple(
        parameters[name]
        for name in required
    )


# --------------
# MAIN FUNCTIONS
# --------------

def model_parameters(f, z, parameters=FULL_FIT):
    """
    Calculate M_S, alpha and beta.

    Parameters
    ----------
    f : float or array_like
        Fraction of the total dark matter in FDM.

    z : float or array_like
        Redshift.

    parameters : dict, optional
        Dictionary containing the nine fitted coefficients.
        Defaults to FULL_FIT.

    Returns
    -------
    M_S : ndarray or float
        Characteristic mass in Msun/h.

    alpha : ndarray or float
        Shape parameter alpha.

    beta : ndarray or float
        Shape parameter beta.
    """

    f = np.asarray(f, dtype=float)
    z = np.asarray(z, dtype=float)

    (m0, mf, mz, a0, af, az, b0, bf, bz) = _get_parameters(parameters)

    log10_Ms = (m0 + mf*f + mz*z)

    alpha = (a0 + af*f + az*z)

    beta = (b0 + bf*f + bz*z)

    M_S = 10.0**log10_Ms

    return M_S, alpha, beta


def log10_suppression_ratio(
    M,
    f,
    z,
    parameters=FULL_FIT,
    check_domain=True,
):
    """
    Evaluate log10 of the MDM-to-CDM HMF suppression ratio.

    The calibrated model is

        R(M, f, z)
        = [1 + (M_S / M)^alpha]^(-beta).

    This function evaluates log10(R) in a numerically stable
    form.

    Parameters
    ----------
    M : float or array_like
        Halo mass M_200c in Msun/h.

    f : float or array_like
        Fraction of the total dark matter in FDM.

    z : float or array_like
        Redshift.

    parameters : dict, optional
        Fitted model coefficients.

    check_domain : bool, optional
        If True, issue warnings when evaluating outside the
        approximate calibration range.

    Returns
    -------
    log10_R : ndarray or float
        log10 of the suppression ratio.
    """

    M = np.asarray(M, dtype=float)

    if np.any(M <= 0):
        raise ValueError("Halo masses must be positive.")

    if check_domain:
        _check_domain(M, f, z)

    (m0, mf, mz, a0, af, az, b0, bf, bz) = _get_parameters(parameters)

    log10_M = np.log10(M)

    log10_Ms = (m0 + mf*np.asarray(f) + mz*np.asarray(z))

    alpha = (a0 + af*np.asarray(f) + az*np.asarray(z))

    beta = (b0 + bf*np.asarray(f) + bz*np.asarray(z))

    # x = log10[(M_S / M)^alpha]
    x = alpha * (log10_Ms - log10_M)

    # Stable evaluation of
    # log10(1 + 10^x)
    # rather than explicitly evaluating 10^x, which can overflow for extreme extrapolations.
    log10_one_plus = (np.logaddexp(0.0, x * np.log(10.0))/ np.log(10.0))

    log10_R = (-beta* log10_one_plus)

    return log10_R


def suppression_ratio(
    M,
    f,
    z,
    parameters=FULL_FIT,
    check_domain=True,
):
    """
    Evaluate the MDM-to-CDM halo mass function ratio.

    Returns

        R = (dn/dlog10M)_MDM/(dn/dlog10M)_CDM.

    Parameters
    ----------
    M : float or array_like
        Halo mass M_200c in Msun/h.

    f : float or array_like
        FDM fraction.

    z : float or array_like
        Redshift.

    parameters : dict, optional
        Fitted model coefficients.

    check_domain : bool, optional
        Warn when evaluating outside the calibration domain.

    Returns
    -------
    R : ndarray or float
        MDM/CDM halo mass function ratio.
    """

    log10_R = log10_suppression_ratio(M=M, f=f, z=z, parameters=parameters, check_domain=check_domain)

    return 10.0**log10_R


def mdm_hmf(
    M,
    dndlog10M_cdm,
    f,
    z,
    parameters=FULL_FIT,
    check_domain=True,
):
    """
    Apply the calibrated transformation to a CDM HMF.

    Parameters
    ----------
    M : array_like
        Halo masses M_200c in Msun/h.

    dndlog10M_cdm : array_like
        CDM halo mass function dn/dlog10(M).

        The units are preserved by the transformation, e.g.
        h^3 Mpc^-3 dex^-1.

    f : float
        FDM fraction.

    z : float
        Redshift.

    parameters : dict, optional
        Fitted model coefficients.

    check_domain : bool, optional
        Warn when evaluating outside the calibration domain.

    Returns
    -------
    dndlog10M_mdm : ndarray
        Predicted MDM halo mass function.
    """

    M = np.asarray(M, dtype=float)
    dndlog10M_cdm = np.asarray(dndlog10M_cdm, dtype=float)

    if M.shape != dndlog10M_cdm.shape:
        raise ValueError(
            "M and dndlog10M_cdm must have the same shape."
        )

    R = suppression_ratio(M=M, f=f, z=z, parameters=parameters, check_domain=check_domain)

    return (R * dndlog10M_cdm)
