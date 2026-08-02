import os
import re
import numpy as np
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import scipy.stats
from scipy.stats import skew, pearsonr
from scipy import stats
from scipy.optimize import differential_evolution
import itertools
import psutil
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import scipy
import scipy.interpolate
import scipy.optimize
import numba
import sys
import gstools
from scipy.spatial import cKDTree
import glob


###############################################################################
# The following functions pertain to fitting the NSRP_spatial parameters #######
# The functions are taken from fitting.py #
def prepare(statistics):
    statistic_ids = sorted(list(set(statistics['statistic_id'])))

    fitting_data = {}
    reference_statistics = []
    weights = []
    gs = []

    # Map string durations into hours
    duration_map = {
        "1H": 1,
        "24H": 24,
        "72H": 72,
        "1M": 720  # assume 30 days * 24 hours; adjust if needed
    }

    for statistic_id in statistic_ids:
        df = statistics.loc[statistics['statistic_id'] == statistic_id].copy()

        duration_raw = df['duration'].values[0]
        # Convert duration to float (if string, map it first)
        if isinstance(duration_raw, str):
            duration = float(duration_map.get(duration_raw, np.nan))
        else:
            duration = float(duration_raw)

        lag = float(df['lag'].values[0])
        threshold = float(df['threshold'].values[0])

        fitting_data[(statistic_id, 'name')] = df['name'].values[0]
        fitting_data[(statistic_id, 'duration')] = duration
        fitting_data[(statistic_id, 'lag')] = lag
        fitting_data[(statistic_id, 'threshold')] = threshold
        fitting_data[(statistic_id, 'df')] = df

        reference_statistics.append(df['value'].astype(float).values)
        weights.append(df['weight'].astype(float).values)
        gs.append(df['gs'].astype(float).values)

    reference_statistics = np.concatenate(reference_statistics)
    weights = np.concatenate(weights)
    gs = np.concatenate(gs)

    return statistic_ids, fitting_data, reference_statistics, weights, gs


def fitting_wrapper(
        parameters, spatial_model, intensity_distribution, statistic_ids, fitting_data, ref_stats, weights, gs,
        all_parameter_names, parameters_to_fit, fixed_parameters, season, nu=None, lamda=None, beta=None, eta=None,
        theta=None, kappa=None
):
    # List of parameters from optimisation can be converted to a dictionary for easier comprehension in analytical
    # property calculations. Fixed parameters can also be included
    parameters_dict = {}
    for parameter_name in all_parameter_names:
        if parameter_name in parameters_to_fit:
            parameters_dict[parameter_name] = parameters[parameters_to_fit.index(parameter_name)]
        else:
            parameters_dict[parameter_name] = fixed_parameters[(season, parameter_name)]

    # If nu is passed then assume that rho is being optimised and gamma should be back-calculated
    # - this will be the second step of fitting a spatial model when the first step is fitting a point model via nu,
    # i.e. typically using a pooled approach to spatial model fitting
    # - fixed parameters should not change in this case - empty dictionary
    # - also need then to add other parameters to dictionary for calculation of analytical properties
    if nu is not None:
        parameters_dict['gamma'] = (2 * np.pi * parameters[0] / nu) ** 0.5
        parameters_dict['lamda'] = lamda
        parameters_dict['beta'] = beta
        parameters_dict['eta'] = eta
        parameters_dict['theta'] = theta
        if intensity_distribution == 'weibull':
            parameters_dict['kappa'] = kappa

    # Calculate properties and objective function
    mod_stats = calculate_analytical_properties(spatial_model, intensity_distribution, parameters_dict, statistic_ids, fitting_data)
    obj_fun = calculate_objective_function(ref_stats, mod_stats, weights, gs)
    return obj_fun


def fit_by_month(unique_months,reference_statistics,spatial_model,intensity_distribution,
    n_workers,all_parameter_names,parameters_to_fit,parameter_bounds,fixed_parameters,
    stage="final",initial_parameters=None,use_pooling=False,):
    """
    Optimise NSRP parameters for each month independently.
    Supports two-stage optimisation if spatial pooling is used
    (nu substitution for rho/gamma).
    """
    results = {}
    fitted_statistics = []

    for month in unique_months:
        print(f"--- Fitting month {month} ---")
        if not isinstance(fixed_parameters, dict):
            if (
                (("rho" in fixed_parameters.columns) and ("gamma" not in fixed_parameters.columns))
                or (("gamma" in fixed_parameters.columns) and ("rho" not in fixed_parameters.columns))
            ):
                raise ValueError("Both rho and gamma must be fixed (or neither fixed).")

        month_ref_stats = reference_statistics.loc[reference_statistics["month"] == month].copy()

        if spatial_model:
            if use_pooling:
                month_ref_stats = month_ref_stats.loc[month_ref_stats["name"] != "cross-correlation_lag0"]
                if "rho" in parameters_to_fit:
                    _spatial_model = False  # stage 1: point model with nu
                else:
                    _spatial_model = True
            else:
                _spatial_model = True
        else:
            _spatial_model = False

        statistic_ids, fitting_data, ref, weights, gs = prepare(month_ref_stats)

        if spatial_model and use_pooling and ("rho" in parameters_to_fit):
            # Stage 1: replace rho, gamma with nu
            _all_parameter_names = [pn for pn in all_parameter_names if pn not in ["rho", "gamma"]] + ["nu"]
            _parameters_to_fit   = [pn for pn in parameters_to_fit if pn not in ["rho", "gamma"]] + ["nu"]
            _fixed_parameters    = fixed_parameters

            # Remove rho & gamma bounds, add nu bounds instead
            rho_min, rho_max     = parameter_bounds[(month, "rho")]
            gamma_min, gamma_max = parameter_bounds[(month, "gamma")]
            nu_min = 2.0 * np.pi * rho_min / gamma_max**2
            nu_max = 2.0 * np.pi * rho_max / gamma_min**2

            _parameter_bounds = [parameter_bounds[(month, p)] for p in parameters_to_fit if p not in ["rho", "gamma"]]
            _parameter_bounds.append((nu_min, nu_max))

        else:
            _all_parameter_names = all_parameter_names
            _parameters_to_fit   = parameters_to_fit
            _fixed_parameters    = fixed_parameters
            _parameter_bounds    = [parameter_bounds[(month, p)] for p in parameters_to_fit]

        x0 = initial_parameters.get(month) if initial_parameters is not None else None
        result = scipy.optimize.differential_evolution(func=fitting_wrapper,bounds=_parameter_bounds,
            args=(
                _spatial_model,
                intensity_distribution,
                statistic_ids,
                fitting_data,
                ref,
                weights,
                gs,
                _all_parameter_names,
                _parameters_to_fit,
                _fixed_parameters,
                month,
            ),
            tol=0.001,
            updating="deferred",
            workers=n_workers,
            x0=x0,
        )

        for idx, pname in enumerate(_parameters_to_fit):
            results[(pname, month)] = result.x[idx]
        results[("converged", month)]            = result.success
        results[("objective_function", month)]   = result.fun
        results[("iterations", month)]           = result.nit
        results[("function_evaluations", month)] = result.nfev

        # -----------------------
        #  Stage 2 optimisation (if pooling with rho/gamma)
        # -----------------------
        if spatial_model and use_pooling and ("rho" in parameters_to_fit):
            month_ref_stats = reference_statistics.loc[
                (reference_statistics["month"] == month)
                & (reference_statistics["name"] == "cross-correlation_lag0")
            ].copy()
            statistic_ids, fitting_data, ref, weights, gs = prepare(month_ref_stats)

            _spatial_model       = True
            _all_parameter_names = ["rho"]
            _parameters_to_fit   = ["rho"]
            _fixed_parameters    = {}
            _parameter_bounds    = [parameter_bounds[(month, "rho")]]

            # Collect required parameters
            nu    = results[("nu", month)]
            lamda = results.get(("lamda", month)) if "lamda" in parameters_to_fit else fixed_parameters.loc[fixed_parameters["month"] == month, "lamda"].values[0]
            beta  = results.get(("beta", month))  if "beta"  in parameters_to_fit else fixed_parameters.loc[fixed_parameters["month"] == month, "beta"].values[0]
            eta   = results.get(("eta", month))   if "eta"   in parameters_to_fit else fixed_parameters.loc[fixed_parameters["month"] == month, "eta"].values[0]
            theta = results.get(("theta", month)) if "theta" in parameters_to_fit else fixed_parameters.loc[fixed_parameters["month"] == month, "theta"].values[0]
            if intensity_distribution == "weibull":
                kappa = results.get(("kappa", month)) if "kappa" in parameters_to_fit else fixed_parameters.loc[fixed_parameters["month"] == month, "kappa"].values[0]
            else:
                kappa = None

            # Stage 2 optimisation for rho
            result2 = scipy.optimize.differential_evolution(
                func=fitting_wrapper,
                bounds=_parameter_bounds,
                args=(
                    _spatial_model,
                    intensity_distribution,
                    statistic_ids,
                    fitting_data,
                    ref,
                    weights,
                    gs,
                    _all_parameter_names,
                    _parameters_to_fit,
                    _fixed_parameters,
                    month,
                    nu,
                    lamda,
                    beta,
                    eta,
                    theta,
                    kappa,
                ),
                tol=0.001,
                updating="deferred",
                workers=n_workers,
                x0=None,
            )

            results[("rho", month)]   = result2.x[0]
            results[("gamma", month)] = (2.0 * np.pi * results[("rho", month)] / results[("nu", month)]) ** 0.5
            # Merge info
            results[("converged", month)]            = result.success and result2.success
            results[("objective_function", month)]   += result2.fun
            results[("iterations", month)]           += result2.nit
            results[("function_evaluations", month)] += result2.nfev
            # Remove intermediate nu
            results.pop(("nu", month))

        # -----------------------
        #  Compute fitted statistics
        # -----------------------
        parameters_dict = {}
        for pname in all_parameter_names:
            if pname == "nu":  # skip intermediate
                continue
            if pname in parameters_to_fit:
                parameters_dict[pname] = results.get((pname, month), np.nan)
            else:
                if isinstance(fixed_parameters, dict):
                    parameters_dict[pname] = fixed_parameters.get((month, pname), np.nan)
                else:
                    val = fixed_parameters.loc[fixed_parameters["month"] == month, pname]
                    parameters_dict[pname] = val.values[0] if len(val) > 0 else np.nan

        dfs = []
        statistic_ids, fitting_data, ref, weights, gs = prepare(
            reference_statistics.loc[reference_statistics["month"] == month]
        )
        mod_stats = calculate_analytical_properties(
            spatial_model, intensity_distribution, parameters_dict, statistic_ids, fitting_data
        )
        for sid in statistic_ids:
            tmp = fitting_data[(sid, "df")].copy()
            dfs.append(tmp)
        df = pd.concat(dfs)
        df["value"] = mod_stats
        df["month"] = month
        fitted_statistics.append(df)

    # -----------------------
    #  Format output
    # -----------------------
    parameters = format_results(
        results, all_parameter_names, parameters_to_fit, fixed_parameters, unique_months, intensity_distribution
    )
    fitted_statistics = pd.concat(fitted_statistics)
    parameters["fit_stage"]       = stage
    fitted_statistics["fit_stage"] = stage

    return parameters, fitted_statistics


def calculate_analytical_properties(spatial_model, intensity_distribution, parameters_dict, statistic_ids, fitting_data):
    # Unpack parameter values common to point and spatial models
    lamda = parameters_dict['lamda']
    beta = parameters_dict['beta']
    eta = parameters_dict['eta']
    theta = parameters_dict['theta']

    # Get or calculate nu
    if not spatial_model:
        nu = parameters_dict['nu']
    else:
        rho = parameters_dict['rho']
        gamma = parameters_dict['gamma']
        nu = 2.0 * np.pi * rho / gamma ** 2.0

    # Shape parameters are only relevant to non-exponential intensity distributions
    if intensity_distribution == 'weibull':
        kappa = parameters_dict['kappa']
    elif intensity_distribution == 'generalised_gamma':
        kappa_1 = parameters_dict['kappa_1']
        kappa_2 = parameters_dict['kappa_2']

    # Calculate raw moments (1-3) of intensity distribution
    moments = []
    for n in [1, 2, 3]:
        if intensity_distribution == 'exponential':
            # mu_1 = 1.0 / (1.0 / theta)
            # mu_2 = 2.0 / (1.0 / theta) ** 2.0
            # mu_3 = 6.0 / (1.0 / theta) ** 3.0
            moments.append(scipy.stats.expon.moment(n, scale=theta))
        elif intensity_distribution == 'weibull':
            moments.append(scipy.stats.weibull_min.moment(n, c=kappa, scale=theta))
        elif intensity_distribution == 'generalised_gamma':
            moments.append(scipy.stats.gengamma.moment(n, a=(kappa_1 / kappa_2), c=kappa_2, scale=theta))
    mu_1, mu_2, mu_3 = moments

    # Main loop to get each required statistic
    statistic_arrays = []
    for statistic_id in statistic_ids:
        name = fitting_data[(statistic_id, 'name')]
        duration = fitting_data[(statistic_id, 'duration')]
        phi = fitting_data[(statistic_id, 'df')]['phi'].values

        if name in ['autocorrelation', 'cross-correlation_lag0']:
            lag = fitting_data[(statistic_id, 'lag')]
            if name == 'cross-correlation_lag0':
                phi2 = fitting_data[(statistic_id, 'df')]['phi2'].values
                distances = fitting_data[(statistic_id, 'df')]['distance'].values
        elif name == 'probability_dry':
            threshold = fitting_data[(statistic_id, 'threshold')]
        if name == 'mean':
            values = calculate_mean(duration, lamda, nu, mu_1, eta, phi)
        elif name == 'variance':
            values = calculate_variance(duration, eta, beta, lamda, nu, mu_1, mu_2, phi)
        elif name == 'skewness':
            values = calculate_skewness(duration, eta, beta, lamda, nu, mu_1, mu_2, mu_3, phi)
        elif name == 'autocorrelation':
            values = calculate_autocorrelation(duration, lag, eta, beta, lamda, nu, mu_1, mu_2, phi)
        elif name == 'probability_dry':
            values = calculate_probability_dry(duration, nu, beta, eta, lamda, phi, threshold)
        elif name == 'cross-correlation_lag0':
            values = calculate_cross_correlation(duration, lag, eta, beta, lamda, nu, mu_1, mu_2, gamma, distances, phi, phi2)
        statistic_arrays.append(values)
    return np.concatenate(statistic_arrays)

def calculate_objective_function(ref, mod, w, sf):
    obj_fun = np.sum((w ** 2 / sf ** 2) * (ref - mod) ** 2)
    return obj_fun

def _mean(h, lamda, nu, mu_X, eta, phi=1):
    """
    Mean of NSRP process.Equation 2.11 in Cowpertwait (1995), which is Equation 5 in Cowpertwait et al. (2002).
    """
    mean_ = phi * h * lamda * nu * mu_X / eta
    return mean_


def calculate_mean(duration, lamda, nu, mu_1, eta, phi):
    mean_ = _mean(duration, lamda, nu, mu_1, eta, phi)
    return mean_


def _covariance_a_b_terms(h, l, eta, beta, lamda, nu, mu_X):
    """A and B terms needed in covariance calculations.

    See Equations 2.12, 2.15 and 2.16 in Cowpertwait (1995).

    """
    # Cowpertwait (1995) equations 2.15 and 2.16
    if l == 0:
        A_hl = 2 * (h * eta + np.exp(-eta * h) - 1) / eta ** 2
        B_hl = 2 * (h * beta + np.exp(-beta * h) - 1) / beta ** 2
    else:
        A_hl = (1 - np.exp(-eta * h)) ** 2 * np.exp(-eta * h * (l - 1)) / eta ** 2
        B_hl = (1 - np.exp(-beta * h)) ** 2 * np.exp(-beta * h * (l - 1)) / beta ** 2

    # Cowpertwait (1995) equation 2.12
    Aij = 0.5 * lamda * beta * nu ** 2 * mu_X ** 2 * ((2 * beta) / ((beta ** 2 - eta ** 2) * (2 * eta)))
    Bij = -0.5 * lamda * beta * nu ** 2 * mu_X ** 2 * (1 / ((beta - eta) * (beta + eta)))

    return A_hl, B_hl, Aij, Bij

def _site_covariance(h, l, eta, beta, lamda, nu, mu_X, var_X, phi=1):
    """Covariance of NSRP process.

    Covariance is calculated as Equation 2.14 in Cowpertwait (1995). This
    requires A and B terms from calculate_A_and_B().

    """
    A_hl, B_hl, Aij, Bij = _covariance_a_b_terms(h, l, eta, beta, lamda, nu, mu_X)

    # Cowpertwait (1995) equation 2.14
    cov = (
            phi ** 2 * (A_hl * Aij + B_hl * Bij) + phi ** 2 * lamda * nu * var_X * A_hl / eta
    )
    return cov

def calculate_variance(duration, eta, beta, lamda, nu, mu_1, mu_2, phi):
    variance = _site_covariance(duration, 0, eta, beta, lamda, nu, mu_1, mu_2, phi)
    return variance

def _skewness_f(eta, beta, h):
    """f-function needed for calculating third central moment.

    Equation 2.10 in Cowpertwait (1998), which is Equation 11 in Cowpertwait
    et al. (2002).

    """
    f = (
        # line 1
        -2 * eta ** 3 * beta ** 2 * np.exp(-eta * h) - 2 * eta ** 3 * beta ** 2 * np.exp(-beta * h)
        + eta ** 2 * beta ** 3 * np.exp(-2 * eta * h) + 2 * eta ** 4 * beta * np.exp(-eta * h)
        # line 2
        + 2 * eta ** 4 * beta * np.exp(-beta * h) + 2 * eta ** 3 * beta ** 2 * np.exp(-(eta + beta) * h)
        - 2 * eta ** 4 * beta * np.exp(-(eta + beta) * h) - 8 * eta ** 3 * beta ** 3 * h
        # line 3
        + 11 * eta ** 2 * beta ** 3 - 2 * eta ** 4 * beta + 2 * eta ** 3 * beta ** 2
        + 4 * eta * beta ** 5 * h + 4 * eta ** 5 * beta * h - 7 * beta ** 5
        # line 4
        - 4 * eta ** 5 + 8 * beta ** 5 * np.exp(-eta * h) - beta ** 5 * np.exp(-2 * eta * h)
        - 2 * h * eta ** 3 * beta ** 3 * np.exp(-eta * h)
        # line 5
        - 12 * eta ** 2 * beta ** 3 * np.exp(-eta * h) + 2 * h * eta * beta ** 5 * np.exp(-eta * h)
        + 4 * eta ** 5 * np.exp(-beta * h)
    )
    return f


def _skewness_g(eta, beta, h):
    """f-function needed for calculating third central moment.

    Equation 2.11 in Cowpertwait (1998), which is Equation 12 in Cowpertwait
    et al. (2002).

    """
    g = (
        # line 1
        12 * eta ** 5 * beta * np.exp(-beta * h) + 9 * eta ** 4 * beta ** 2 + 12 * eta * beta ** 5 * np.exp(-eta * h)
        + 9 * eta ** 2 * beta ** 4
        # line 2
        + 12 * eta ** 3 * beta ** 3 * np.exp(-(eta + beta) * h) - eta ** 2 * beta ** 4 * np.exp(-2 * eta * h)
        - 12 * eta ** 3 * beta ** 3 * np.exp(-beta * h) - 9 * eta ** 5 * beta
        # line 3
        - 9 * eta * beta ** 5 - 3 * eta * beta ** 5 * np.exp(-2 * eta * h)
        - eta ** 4 * beta ** 2 * np.exp(-2 * beta * h) - 12 * eta ** 3 * beta ** 3 * np.exp(-eta * h)
        # line 4
        + 6 * eta ** 5 * beta ** 2 * h - 10 * beta ** 4 * eta ** 3 * h + 6 * beta ** 5 * eta ** 2 * h
        - 10 * beta ** 3 * eta ** 4 * h + 4 * beta ** 6 * eta * h
        # line 5
        - 8 * beta ** 2 * eta ** 4 * np.exp(-beta * h) + 4 * beta * eta ** 6 * h + 12 * beta ** 3 * eta ** 3
        - 8 * beta ** 4 * eta ** 2 * np.exp(-eta * h) - 6 * eta ** 6
        # line 6
        - 6 * beta ** 6 - 2 * eta ** 6 * np.exp(-2 * beta * h) - 2 * beta ** 6 * np.exp(-2 * eta * h)
        + 8 * eta ** 6 * np.exp(-beta * h)
        # line 7
        + 8 * beta ** 6 * np.exp(-eta * h) - 3 * beta * eta ** 5 * np.exp(-2 * beta * h)
    )
    return g


def _third_central_moment(
        h, eta, beta, lamda, nu, mu_X, var_X, X_mom3
):
    """Third central moment of NSRP process.

    Equation 2.9 in Cowpertwait (1998), which is Equation 10 in Cowpertwait
    et al. (2002). Requires f-function and g-function from skewness_f() and
    skewness_g(), respectively.

    """
    f = _skewness_f(eta, beta, h)
    g = _skewness_g(eta, beta, h)

    # Cowpertwait (1998) equation 2.9
    skew = (
        # line 1
        6 * lamda * nu * X_mom3 * (eta * h - 2 + eta * h * np.exp(-eta * h) + 2 * np.exp(-eta * h)) / eta ** 4
        # line 2
        + 3 * lamda * mu_X * var_X * nu ** 2 * f
        # line 3
        / (2 * eta ** 4 * beta * (beta ** 2 - eta ** 2) ** 2) + lamda * mu_X ** 3
        # line 4
        * nu ** 3 * g
        # line 5
        / (2 * eta ** 4 * beta * (eta ** 2 - beta ** 2) * (eta - beta) * (2 * beta + eta) * (beta + 2 * eta))
    )
    return skew



def calculate_skewness(duration, eta, beta, lamda, nu, mu_1, mu_2, mu_3, phi):
    unscaled_variance = _site_covariance(duration, 0, eta, beta, lamda, nu, mu_1, mu_2, phi * 0.0 + 1.0)
    third_moment = _third_central_moment(duration, eta, beta, lamda, nu, mu_1, mu_2, mu_3)
    skewness = third_moment / (unscaled_variance ** 0.5) ** 3
    return skewness


def calculate_autocorrelation(duration, lag, eta, beta, lamda, nu, mu_1, mu_2, phi):
    variance = _site_covariance(duration, 0, eta, beta, lamda, nu, mu_1, mu_2, phi)
    lag_covariance = _site_covariance(duration, lag, eta, beta, lamda, nu, mu_1, mu_2, phi)
    autocorrelation = lag_covariance / variance
    return autocorrelation


def _omega(beta, t, eta):
    """Omega term in Equation 2.17 in Cowpertwait (1995).

    Probability that a cell overlapping point m with arrival time in (0, t)
    terminates before t. Same as Equation 2.15 in Cowpertwait (1994).

    """
    omega = 1 - beta * (np.exp(-beta * t) - np.exp(-eta * t)) / ((eta - beta) * (1 - np.exp(-beta * t)))
    return omega

def _probability_zero_t_0(t, nu, beta, eta):
    """Probability of no rain in (0, t).

    Equation 2.18 in Cowpertwait (1995) but setting t=0 and h=t.

    Returns 1 minus the probability, as this is what is needed to find the dry
    probability using Equation 2.19 in Cowpertwait (1995).

    """
    omega_ = _omega(beta, t, eta)
    p = np.exp(-nu + nu * np.exp(-beta * (0 + t)) + omega_ * nu * (1 - np.exp(-beta * 0)))
    return 1 - p



def _probability_zero_h_t(t, h, nu, beta, eta):
    """Probability of no rain in (t, t+h) due to a storm origin at time zero.

    Equation 2.18 in Cowpertwait (1995). I.e. differs from Cowpertwait (1994),
    as number of cells per storm is a Poisson random variable, whereas
    Cowpertwait (1994) used a geometric distribution.

    Returns 1 minus the probability, as this is what is needed to find the dry
    probability using Equation 2.19 in Cowpertwait (1995).

    """
    omega_ = _omega(beta, t, eta)

    # Cowpertwait (1995) equation 2.18
    p = np.exp(-nu + nu * np.exp(-beta * (t + h)) + omega_ * nu * (1 - np.exp(-beta * t)))
    return 1 - p

def _probability_dry(h, nu, beta, eta, lamda):
    """Probability dry (equal to zero) for NSRP process.

    Equation 2.19 in Cowpertwait (1995).

    """
    term1, term1_error = scipy.integrate.quad(_probability_zero_h_t, 0, np.inf, args=(h, nu, beta, eta))
    term2, term2_error = scipy.integrate.quad(_probability_zero_t_0, 0, h, args=(nu, beta, eta))
    p = np.exp(-lamda * term1 - lamda * term2)
    return p


def _probability_dry_correction(h, threshold, uncorr_pdry):
    """Estimation of dry probability for non-zero thresholds.

    Following Section 4.3 in Burton et al. (2008). Options are only for 24hr
    duration (thresholds of 0.2 or 1.0 mm) or 1hr duration (thresholds of 0.1 or
    0.2 mm).

    """
    if h == 24:

        # Burton et al. (2008) equation 8
        if threshold == 1.0:
            if 0.15 <= uncorr_pdry <= 0.75:
                corr_pdry = 0.05999 + 1.603 * uncorr_pdry - 0.8138 * uncorr_pdry ** 2
            elif uncorr_pdry < 0.15:
                dx = 0.15
                dy = 0.2821
                m = dy / dx
                corr_pdry = m * uncorr_pdry
            elif uncorr_pdry > 0.75:
                dx = 0.75
                dy = 0.8045
                m = dy / dx
                corr_pdry = m * uncorr_pdry

        # Burton et al. (2008) equation 9
        elif threshold == 0.1:
            corr_pdry = 0.219067 + 0.789442 * uncorr_pdry
            
        elif threshold == 0.2:
            if 0.2 <= uncorr_pdry <= 0.75:
                corr_pdry = 0.007402 + 1.224 * uncorr_pdry - 0.2908 * uncorr_pdry ** 2
            elif uncorr_pdry < 0.2:
                dx = 0.2
                dy = 0.2405
                m = dy / dx
                corr_pdry = m * uncorr_pdry
            elif uncorr_pdry > 0.75:
                dx = 0.75
                dy = 0.7617
                m = dy / dx
                corr_pdry = m * uncorr_pdry

    elif h == 1:

        # Bias correction equation derived for 457 gauges
        if threshold == 0.1:
            corr_pdry = 0.219067 + 0.789442 * uncorr_pdry

        # Burton et al. (2008) equation 11
        elif threshold == 0.2:
            corr_pdry = 0.239678 + 0.758837 * uncorr_pdry
        corr_pdry = max(corr_pdry, 0.0)
        corr_pdry = min(corr_pdry, 1.0)

    return corr_pdry


def calculate_probability_dry(duration, nu, beta, eta, lamda, phi, threshold=None):
    probability_dry = _probability_dry(duration, nu, beta, eta, lamda)
    if threshold is not None:
        probability_dry = _probability_dry_correction(duration, threshold, probability_dry)
    probability_dry = phi * 0.0 + probability_dry
    return probability_dry


def _probability_overlap_integral_expression(y, gamma, d):
    # Cowpertwait et al. (2002) equation 8 / Cowpertwait (2010) page 3
    expr = ((gamma * d) / (2 * np.cos(y)) + 1) * np.exp((-gamma * d) / (2 * np.cos(y)))
    return expr


def _cross_covariance(h, l, eta, beta, lamda, nu, mu_X, var_X, gamma, d, phi1=1, phi2=1):
    # Cell overlap probability
    integral_term, error = scipy.integrate.quad(
        _probability_overlap_integral_expression, 0, np.pi / 2, args=(gamma, d)
    )
    overlap_probability = 2 / np.pi * integral_term

    a_hl, b_hl, aij, bij = _covariance_a_b_terms(h, l, eta, beta, lamda, nu, mu_X)

    # Cowpertwait (1995) equation 2.24
    cov = (
            phi1 * phi2 * (a_hl * aij + b_hl * bij)
            + phi1 * phi2 * lamda * overlap_probability * nu * var_X * a_hl / eta
    )

    return cov


def calculate_cross_correlation(
        duration, lag, eta, beta, lamda, nu, mu_1, mu_2, gamma, distances, phi1, phi2
):
    # For lags > 0 then need to calculate unscaled_variance for both lag=0 and lag=lag
    # then use these appropriately below - IMPLEMENT THIS
    if lag != 0:
        raise ValueError('Cross-correlation not yet implemented for lags > 0')
    else:
        pass

    cross_correlations = []

    unscaled_variance = _site_covariance(duration, lag, eta, beta, lamda, nu, mu_1, mu_2, 1)

    for idx in range(phi1.shape[0]):
        variance1 = unscaled_variance * phi1[idx] ** 2
        variance2 = unscaled_variance * phi2[idx] ** 2
        covariance = _cross_covariance(duration, lag, eta, beta, lamda, nu, mu_1, mu_2, gamma, distances[idx], phi1[idx],phi2[idx])  # h, l, eta, beta, lamda, nu, mu_X, var_X, gamma, d, phi1=1, phi2=1
        cross_correlation = covariance / (variance1 ** 0.5 * variance2 ** 0.5)
        cross_correlations.append(cross_correlation)

    return np.asarray(cross_correlations)


def format_results(results, all_parameter_names, parameters_to_fit, fixed_parameters, unique_months, intensity_distribution):
    """
    Convert the results dictionary into a DataFrame of fitted parameters.
    Handles fixed vs fitted parameters and includes diagnostics.
    Column names are aligned case-insensitively to avoid duplicates.
    """

    rows = []
    for month in unique_months:
        row = {
            "month": month,
            "fit_stage": "final",
            "intensity_distribution": intensity_distribution,
        }

        # Parameters
        for pname in all_parameter_names:
            if pname == "nu":  
                continue  # skip nu in final output
            try:
                row[pname] = results[(pname, month)]
            except KeyError:
                if isinstance(fixed_parameters, dict) and (month, pname) in fixed_parameters:
                    row[pname] = fixed_parameters[(month, pname)]
                else:
                    row[pname] = np.nan

        # Diagnostics
        row["converged"] = results.get(("converged", month), False)
        row["objective_function"] = results.get(("objective_function", month), np.nan)
        row["iterations"] = results.get(("iterations", month), np.nan)
        row["function_evaluations"] = results.get(("function_evaluations", month), np.nan)

        rows.append(row)

    df = pd.DataFrame(rows)

    # Desired column order (case-insensitive)
    col_order = [
        "fit_stage", "month",
        "lamda", "beta", "rho", "eta", "gamma", "theta", "kappa",
        "converged", "objective_function", "iterations", "function_evaluations",
        "intensity_distribution"
    ]

    # Build mapping {lowercase: actual_column_name}
    existing_cols = {c.lower(): c for c in df.columns}
    ordered_cols = [existing_cols[c] for c in col_order if c in existing_cols]
    extra_cols = [c for c in df.columns if c not in ordered_cols]
    return df[ordered_cols + extra_cols]





unique_months = list(range(1,13))
all_parameter_names = ['lamda','beta','eta','theta','kappa','rho','gamma','nu']
parameters_to_fit   = ['lamda','beta','eta','theta','kappa','rho','gamma','nu']
fixed_parameters = {}
parameter_bounds = {
    (m, param): bounds
    for m in range(1, 13)
    for param, bounds in {
        'lamda': (0.001, 0.05),
        'beta': (0.02, 0.5),
        'rho':(0.0001,0.05),
        'eta': (0.1, 12),
        'gamma': (0.01, 500),
        'theta': (0.25, 100),
        'nu': (0.1, 30),
        'kappa': (0.5,1)
    }.items()}



os.chdir('/home/users/azhar199/DATA/NEWDATA/GRIDDED_WG_QC/CORRELATION/PerturbStats3/')
FILES = sorted(glob.glob('*.csv'))

def fit_month_task(month):
    return fit_by_month(
        unique_months=[month],
        reference_statistics=reference_statistics,
        spatial_model=True,
        intensity_distribution='weibull',
        n_workers=1,
        all_parameter_names=all_parameter_names,
        parameters_to_fit=parameters_to_fit,
        parameter_bounds=parameter_bounds,
        fixed_parameters=fixed_parameters,
        stage='final',
        initial_parameters=None,
        use_pooling=False,
    )


def GetParams(FILEREF):

    global reference_statistics
    reference_statistics = pd.read_csv(FILEREF)
    with Pool(processes=4) as pool:
        results = pool.map(fit_month_task, unique_months)

    parameters_df = pd.concat([r[0] for r in results],axis=0).reset_index(drop=True)
    return parameters_df

if __name__ == "__main__":
    idx = int(sys.argv[1])
    FN = FILES[idx]
    outdir = "/home/users/azhar199/DATA/NEWDATA/GRIDDED_WG_QC/CORRELATION/SPATIAL_PARAMETERS_FIN"
    os.makedirs(outdir,exist_ok=True)
    RPDF = GetParams(FN)
    ref_path = os.path.join(outdir, f"PARAMS_{FN}.csv")
    RPDF.to_csv(ref_path,index=False)
