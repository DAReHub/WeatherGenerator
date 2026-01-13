## This python script for NSRP based rainfall simulation for HadUK which has daily weather variables

import os
import pandas as pd
import numpy as np
import datetime
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
from scipy.stats import skew
import statsmodels.api as sm
import scipy.stats
from scipy.optimize import differential_evolution
from multiprocessing import Pool
import scipy.special
import numba
from concurrent.futures import ProcessPoolExecutor, as_completed
from netCDF4 import Dataset
import glob
import matplotlib.pyplot as plt
import calendar
from matplotlib.lines import Line2D
import seaborn as sns
from scipy.stats import genextreme as gev


## Data/time series preparation stage begins from here ##
def prepare_point_timeseries(df, season_definitions, completeness_threshold, durations, outlier_method,
        maximum_relative_difference, maximum_alterations):
    """
    Prepare point timeseries for analysis.

    Steps are: (1) subset on reference calculation period, (2) define seasons for grouping, (3) applying any trimming
    or clipping to reduce the influence of outliers, and (4) aggregating timeseries to required durations.

    """
    # Check valid or nan  # TODO: Revisit if this function gets used for non-precipitation variables
    df.loc[df['value'] < 0.0] = np.nan

    # Apply season definitions and make a running UID for season that goes up by one at each change in season
    # through the time series. Season definitions are needed to identify season completeness but also to apply
    # trimming or clipping
    df['season'] = df.index.month.map(season_definitions)
    df['season_uid'] = df['season'].ne(df['season'].shift()).cumsum()

    # Mask periods not meeting data completeness threshold (close approximation). There is an assumption of at
    # least one complete version of each season in dataframe (where complete means that nans may be present - i.e.
    # fine unless only a very short (< 1 year) record is passed in)
    if df['value'].isnull().any():
        df['season_count'] = df.groupby('season_uid')['value'].transform('count')
        df['season_size'] = df.groupby('season_uid')['value'].transform('size')
        df['season_size'] = df.groupby('season')['season_size'].transform('median')
        df['completeness'] = df['season_count'] / df['season_size'] * 100.0
        df['completeness'] = np.where(df['completeness'] > 100.0, 100.0, df['completeness'])
        df.loc[df['completeness'] < completeness_threshold, 'value'] = np.nan
        df = df.loc[:, ['season', 'value']]

    # Apply trimming or clipping season-wise
    if outlier_method == 'trim':
        df['value'] = df.groupby('season')['value'].transform(
            trim_array(maximum_relative_difference, maximum_alterations)
        )
    elif outlier_method == 'clip':
        df['value'] = df.groupby('season')['value'].transform(
            clip_array(maximum_relative_difference, maximum_alterations)
        )

    # Find timestep and convert from datetime to period index if needed
    if not isinstance(df.index, pd.PeriodIndex):
        datetime_difference = df.index[1] - df.index[0]
    else:
        datetime_difference = df.index[1].to_timestamp() - df.index[0].to_timestamp()
    timestep_length = int(datetime_difference.days * 24) + int(datetime_difference.seconds / 3600)  # hours
    period = str(timestep_length) + 'H'  # TODO: Sort out sub-hourly timestep
    if not isinstance(df.index, pd.PeriodIndex):
        df = df.to_period(period)

    duration_hours = []
    for duration in durations:
        duration_units = duration[-1]
        if duration_units == 'H':
            duration_hours.append(int(duration[:-1]))
        elif duration_units == 'D':
            duration_hours.append(int(duration[:-1]) * 24)
        elif duration_units == 'M':
            duration_hours.append(31 * 24)
    duration_hours = np.asarray(duration_hours)
    sorted_durations = np.asarray(durations)[np.argsort(duration_hours)]

    # Aggregate timeseries to required durations
    dfs = {}
    for duration in sorted_durations:
        # resample_code = str(int(duration)) + 'H'  # TODO: Check/add sub-hourly
        resample_code = duration
        duration_units = duration[-1]
        if duration_units == 'H':
            duration_hours = int(duration[:-1])
        elif duration_units == 'D':
            duration_hours = int(duration[:-1]) * 24
        elif duration_units == 'M':
            duration_hours = 31 * 24

        # Final day needed for a given aggregation
        # - relies on multiples of one day if duration exceeds 24 hours
        # - constrained to monthly
        # - maximum duration of 28 days(?)
        if duration_hours > 24:
            duration_days = int(duration_hours / 24)

            # Interim aggregation to daily to see if it speeds things up
            if '24H' in durations:
                df1 = dfs['24H'].copy()
            elif '1D' in durations:
                df1 = dfs['1D'].copy()

            n_groups = int(np.ceil(31 / duration_days))
            df1['group'] = -1
            for group in range(n_groups):
                if duration_units != 'M':
                    df1['group'] = np.where(df1.index.day >= group * duration_days + 1, group, df1['group'])
                else:
                    # df1['month'] = df1.index.month
                    # df1['group'] = df1['month'].ne(df1['month'].shift()).cumsum()
                    # df1.drop(columns=['month'], inplace=True)
                    df1['group'] = 0

            # df1 = df.groupby([df.index.year, df.index.month, 'group'])['value'].agg(['sum', 'count'])
            df1 = df1.groupby([df1.index.year, df1.index.month, 'group'])['value'].agg(['sum', 'count'])
            if df1.index.names[0] == 'datetime':  # !221025 - for dfs coming from shuffling (fitting delta)
                df1.index.rename(['level_0', 'level_1', 'group'], inplace=True)
            df1.reset_index(inplace=True)
            df1['day'] = df1['group'] * duration_days + 1
            df1.rename(columns={'level_0': 'year', 'level_1': 'month'}, inplace=True)
            df1['datetime'] = pd.to_datetime(df1[['year', 'month', 'day']])
            df1.drop(columns=['year', 'month', 'day', 'group'], inplace=True)
            df1.set_index('datetime', inplace=True)
            # print(df1)
        else:
            df1 = df['value'].resample(resample_code, closed='left', label='left').agg(['sum', 'count'])
            # df2 = df['value'].resample(resample_code, closed='left', label='left').count()

        # df1 = df['value'].resample(resample_code, closed='left', label='left').sum()
        # df2 = df['value'].resample(resample_code, closed='left', label='left').count()

        # Remove data below a duration-dependent completeness
        if duration_hours <= 24:  # TODO: Remove hardcoding of timestep requiring complete data and completeness threshold?
            expected_count = int(duration_hours / timestep_length)
        else:
            expected_count = ((duration_hours / timestep_length) / 24) * 0.9  # TODO: Remove hardcoding - user option
        # df1.values[df2.values < expected_count] = np.nan  # duration
        df1.rename(columns={'sum': 'value'}, inplace=True)
        df1.loc[df1['count'] < expected_count, 'value'] = np.nan
        df1.drop(columns=['count'], inplace=True)
        df1.sort_index(inplace=True)
        df1['season'] = df1.index.month.map(season_definitions)
        dfs[duration] = df1
        dfs[duration] = dfs[duration][dfs[duration]['value'].notnull()]
        
    return dfs


def trim_array(max_relative_difference, max_removals):
    def f(x):
        y = x.copy()
        removals = 0
        while True:
            y_max = np.max(y)
            y_max_count = np.sum(y == y_max)
            y_next_largest = np.max(y[y < y_max])
            if y_max / y_next_largest > max_relative_difference:
                if removals + y_max_count <= max_removals:
                    y = y[y < y_max]
                    removals += y_max_count
                else:
                    break
            else:
                break
        return y  # , removals
    return f


def clip_array(max_relative_difference, max_clips):
    # - assuming working with zero-bounded values
    def f(x):
        y = x.copy()
        clips = 0
        clip_flag = -999
        while True:
            y_max = np.max(y)
            y_max_count = np.sum(y == y_max)
            y_next_largest = np.max(y[y < y_max])
            if y_max / y_next_largest > max_relative_difference:
                if clips + y_max_count <= max_clips:
                    y[y == y_max] = clip_flag
                    clips += y_max_count
                else:
                    break
            else:
                break
        y[y == clip_flag] = np.max(y)
        return y  # , clips
    return f


## We get reference statistics from this stage and 
## data preparation stage ends here 

def GetMonthStats(ListofDFs,WET_THRESHOLD):
    '''
    This function requires outputs from the functions PrepareTimeSeriesPoint
    and nested_dictionary_to_dataframe
    '''
    dc = {
            1: {'weight': 1.0, 'duration': '1H', 'name': 'variance'},
            2: {'weight': 2.0, 'duration': '1H', 'name': 'skewness'},
            3: {'weight': 7.0, 'duration': '1H', 'name': f'probability_dry_{WET_THRESHOLD}mm', 'threshold': WET_THRESHOLD},
            4: {'weight': 6.0, 'duration': '24H', 'name': 'mean'},
            5: {'weight': 2.0, 'duration': '24H', 'name': 'variance'},
            6: {'weight': 3.0, 'duration': '24H', 'name': 'skewness'},
            7: {'weight': 7.0, 'duration': '24H', 'name': f'probability_dry_{WET_THRESHOLD}mm', 'threshold': WET_THRESHOLD},
            8: {'weight': 6.0, 'duration': '24H', 'name': 'autocorrelation_lag1', 'lag': 1},
            9: {'weight': 3.0, 'duration': '72H', 'name': 'variance'},
            10: {'weight': 0.0, 'duration': '1M', 'name': 'variance'},
        }
    id_name = 'statistic_id'
    non_id_columns = ['name', 'duration', 'lag', 'threshold', 'weight']


    def nested_dictionary_to_dataframe(dc, id_name, non_id_columns):
        ids = sorted(list(dc.keys()))
        data = {}
        for non_id_column in non_id_columns:
            data[non_id_column] = []
            for id_ in ids:
                values = dc[id_]
                data[non_id_column].append(
                    values[non_id_column] if non_id_column in values.keys() else 'NA'
                )
        dc1 = {}
        dc1[id_name] = ids
        for non_id_column in non_id_columns:
            dc1[non_id_column] = data[non_id_column]
        df = pd.DataFrame(dc1)
        return df
        
    statistic_definitions = nested_dictionary_to_dataframe(dc, id_name, non_id_columns)
    statistic_definitions = statistic_definitions[statistic_definitions['duration']!='1H'].reset_index(drop = True)
    
    statistic_definitions[statistic_definitions['duration']=='24H'].name
    statistic_definitions[statistic_definitions['duration']=='72H'].name
    statistic_definitions[statistic_definitions['duration']=='1M'].name

    ListofDFs['24H']['Month'] = [ListofDFs['24H'].index[i].month for i in np.arange(0,ListofDFs['24H'].shape[0],1)]
    ListofDFs['24H']['Year'] = [ListofDFs['24H'].index[i].year for i in np.arange(0,ListofDFs['24H'].shape[0],1)]

    ListofDFs['72H']['Month'] = [ListofDFs['72H'].index[i].month for i in np.arange(0,ListofDFs['72H'].shape[0],1)]
    ListofDFs['72H']['Year'] = [ListofDFs['72H'].index[i].year for i in np.arange(0,ListofDFs['72H'].shape[0],1)]

    ListofDFs['1M']['Month'] = [ListofDFs['1M'].index[i].month for i in np.arange(0,ListofDFs['1M'].shape[0],1)]
    ListofDFs['1M']['Year'] = [ListofDFs['1M'].index[i].year for i in np.arange(0,ListofDFs['1M'].shape[0],1)]

    MEAN_24H = [np.nanmean(ListofDFs['24H']['value'][ListofDFs['24H']['Month']==i]) for i in range(1,13)]
    VAR_24H = [np.nanvar(ListofDFs['24H']['value'][ListofDFs['24H']['Month']==i]) for i in range(1,13)]
    SKEW_24H = [skew(ListofDFs['24H']['value'][ListofDFs['24H']['Month'] == i]) for i in range(1, 13)]
    PROB_24H = [len(ListofDFs['24H']['value'][(ListofDFs['24H']['Month'] == i) & (ListofDFs['24H']['value'] < WET_THRESHOLD)].values)/len(ListofDFs['24H']['value'][ListofDFs['24H']['Month']==i]) for i in range(1,13)]

    def getacf(MONTHNUM):
        df=pd.DataFrame({'x': ListofDFs['24H']['value'][ListofDFs['24H']['Month']==MONTHNUM], 'x_lag': ListofDFs['24H']['value'][ListofDFs['24H']['Month']==MONTHNUM].shift(1)})
        df.dropna(inplace=True)
        acf,pval = scipy.stats.pearsonr(df['x'], df['x_lag'])
        return acf
 
    ACF_24H=[getacf(i) for i in range(1,13)]
    VAR_72H = [np.nanvar(ListofDFs['72H']['value'][ListofDFs['72H']['Month']==i]) for i in range(1,13)]
    VAR_1M = [np.nanvar(ListofDFs['1M']['value'][ListofDFs['1M']['Month']==i]) for i in range(1,13)]

    # standardization of the statistics
    STAN_MEAN_24H = np.mean([np.nanmean(ListofDFs['24H']['value'][ListofDFs['24H']['Year']==i]) for i in np.arange(min(ListofDFs['24H']['Year']),max(ListofDFs['24H']['Year'])+1,1)])
    STAN_VAR_24H = np.mean([np.nanvar(ListofDFs['24H']['value'][ListofDFs['24H']['Year']==i]) for i in np.unique(ListofDFs['24H']['Year'])])
    STAN_SKEW_24H = np.mean([skew(ListofDFs['24H']['value'][ListofDFs['24H']['Year']==i], nan_policy='omit') for i in np.unique(ListofDFs['24H']['Year'])])
    STAN_VAR_72H = np.mean([np.nanvar(ListofDFs['72H']['value'][ListofDFs['72H']['Year']==i]) for i in np.unique(ListofDFs['72H']['Year'])])
    STAN_VAR_1M = np.mean([np.nanvar(ListofDFs['1M']['value'][ListofDFs['1M']['Year']==i]) for i in np.unique(ListofDFs['1M']['Year'])])

    STAT =  pd.DataFrame({'statistic_id':np.repeat(range(1,statistic_definitions.shape[0]+1),12),
       'name':np.repeat(statistic_definitions['name'],12),
       'duration':np.repeat(statistic_definitions['duration'],12),
       'month':np.tile(range(1,13),statistic_definitions.shape[0]),
       'value':np.concatenate((MEAN_24H,VAR_24H,SKEW_24H,PROB_24H,ACF_24H,VAR_72H,VAR_1M)),
       'weight':np.repeat(statistic_definitions['weight'],12),
       'gs':np.concatenate((np.repeat(STAN_MEAN_24H,12),np.repeat(STAN_VAR_24H,12),np.repeat(STAN_SKEW_24H,12),np.repeat(1,24),np.repeat(STAN_VAR_72H,12),np.repeat(STAN_VAR_1M,12))),
       'phi':np.repeat(1,12*statistic_definitions.shape[0])})
    
    STAT.loc[STAT['name'].str.contains('lag1'), 'lag'] = 1
    STAT.loc[STAT['name'].str.contains('probability_dry'), 'threshold'] = WET_THRESHOLD
    return STAT
    

##  Getting reference statistics stage ends here and 
##  fitting stage begins from here 

def prepare(statistics):
    statistic_ids = sorted(list(set(statistics['statistic_id'])))

    fitting_data = {}
    reference_statistics = []
    weights = []
    gs = []
    for statistic_id in statistic_ids:
        df = statistics.loc[statistics['statistic_id'] == statistic_id].copy()

        fitting_data[(statistic_id, 'name')] = df['name'].values[0]
        fitting_data[(statistic_id, 'duration')] = df['duration'].values[0]
        fitting_data[(statistic_id, 'lag')] = df['lag'].values[0]
        fitting_data[(statistic_id, 'threshold')] = df['threshold'].values[0]
        fitting_data[(statistic_id, 'df')] = df

        reference_statistics.append(df['value'].values)
        weights.append(df['weight'].values)
        gs.append(df['gs'].values)

    reference_statistics = np.concatenate(reference_statistics)
    weights = np.concatenate(weights)
    gs = np.concatenate(gs)

    return statistic_ids, fitting_data, reference_statistics, weights, gs


def fitting_wrapper_point(
        parameters, intensity_distribution, statistic_ids, fitting_data,
        ref_stats, weights, gs, all_parameter_names, parameters_to_fit,
        fixed_parameters, month
):
    """
    Objective function for monthly point NSRP fitting.
    Compares modelled statistics to observed statistics for one month.
    """
    # Build full parameter dictionary
    parameters_dict = {}
    for pname in all_parameter_names:
        if pname in parameters_to_fit:
            parameters_dict[pname] = parameters[parameters_to_fit.index(pname)]
        else:
            parameters_dict[pname] = fixed_parameters[(month, pname)]

    # Calculate model statistics
    mod_stats = calculate_analytical_properties(
        spatial_model=False,
        intensity_distribution=intensity_distribution,
        parameters_dict=parameters_dict,
        statistic_ids=statistic_ids,
        fitting_data=fitting_data
    )

    # Return weighted and scaled error
    return calculate_objective_function(ref_stats, mod_stats, weights, gs)

def fit_by_month_point(
        unique_months, reference_statistics, intensity_distribution,
        all_parameter_names, parameters_to_fit, parameter_bounds,
        fixed_parameters, n_workers=1, stage='final', initial_parameters=None
):
    """
    Fit NSRP parameters for each month independently (point model only).
    """
    results = {}
    fitted_statistics = []

    for month in unique_months:
        # Filter reference statistics for current month
        month_ref_stats = reference_statistics.loc[reference_statistics['month'] == month].copy()

        # Prepare fitting data
        statistic_ids, fitting_data, ref, weights, gs = prepare(month_ref_stats)

        # Parameter bounds for this month
        bounds = [parameter_bounds[(month, p)] for p in parameters_to_fit]

        # Initial guess if available
        x0 = initial_parameters[month] if initial_parameters is not None else None

        # Run optimisation
        result = differential_evolution(
            func=fitting_wrapper_point,
            bounds=bounds,
            args=(intensity_distribution, statistic_ids, fitting_data, ref,
                  weights, gs, all_parameter_names, parameters_to_fit,
                  fixed_parameters, month),
            tol=0.001,
            updating='deferred',
            workers=n_workers,
            x0=x0
        )

        # Store results for this month
        for idx, pname in enumerate(parameters_to_fit):
            results[(pname, month)] = result.x[idx]
        results[('converged', month)] = result.success
        results[('objective_function', month)] = result.fun
        results[('iterations', month)] = result.nit
        results[('function_evaluations', month)] = result.nfev

        # Build parameter dictionary for fitted stats
        parameters_dict = {}
        for pname in all_parameter_names:
            if pname in parameters_to_fit:
                parameters_dict[pname] = results[(pname, month)]
            else:
                parameters_dict[pname] = fixed_parameters[(month, pname)]

        # Calculate fitted statistics
        mod_stats = calculate_analytical_properties(
            spatial_model=False,
            intensity_distribution=intensity_distribution,
            parameters_dict=parameters_dict,
            statistic_ids=statistic_ids,
            fitting_data=fitting_data
        )

        df_stats = month_ref_stats.copy()
        df_stats['value'] = mod_stats
        df_stats['month'] = month
        fitted_statistics.append(df_stats)

    # Format parameter table
    parameters_df = format_results(results, all_parameter_names, parameters_to_fit, fixed_parameters, unique_months,intensity_distribution)
    parameters_df['fit_stage'] = stage

    # Combine fitted statistics
    fitted_statistics = pd.concat(fitted_statistics)
    fitted_statistics['fit_stage'] = stage

    return parameters_df, fitted_statistics

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
            moments.append(scipy.stats.expon.moment(n, scale=theta))
        elif intensity_distribution == 'weibull':
            moments.append(scipy.stats.weibull_min.moment(n, c=kappa, scale=theta))
        elif intensity_distribution == 'generalised_gamma':
            moments.append(scipy.stats.gengamma.moment(n, a=(kappa_1 / kappa_2), c=kappa_2, scale=theta))
    mu_1, mu_2, mu_3 = moments

    # Duration string → numeric hours mapping
    duration_map = {
        '1h': 1.0,
        '24h': 24.0,
        '72h': 72.0,
        '1m': 24.0 * 30.0  # approx 30 days per month, adjust if needed
    }

    statistic_arrays = []
    for statistic_id in statistic_ids:
        name = fitting_data[(statistic_id, 'name')]
        duration_str = fitting_data[(statistic_id, 'duration')]
        # normalize strings to lower-case to match robustly
        name_l = str(name).lower()
        dur_l = str(duration_str).lower()

        # convert duration to numeric hours
        if dur_l in duration_map:
            duration_val = duration_map[dur_l]
        else:
            # try to parse numeric prefix if user used formats like '1H' or '24H' etc.
            try:
                if dur_l.endswith('h'):
                    duration_val = float(dur_l[:-1])
                elif dur_l.endswith('m') and len(dur_l) > 1:  # '1M' treat as month
                    duration_val = duration_map.get('1m', 24.0 * 30.0)
                else:
                    duration_val = float(dur_l)
            except Exception:
                raise ValueError(f"Unknown duration string: {duration_str}")

        phi = np.ones(len(fitting_data[(statistic_id, 'df')]))

        # detect statistic types using substring matching
        is_autocorr = 'autocorrelation' in name_l
        is_crosscorr = 'cross-correlation' in name_l or 'cross_correlation' in name_l or 'cross correlation' in name_l or 'cross' in name_l and 'correlation' in name_l
        is_prob = 'probability_dry' in name_l or name_l.startswith('probability')
        is_mean = 'mean' in name_l and not is_prob
        is_variance = 'variance' in name_l
        is_skew = 'skew' in name_l

        if is_autocorr or is_crosscorr:
            lag = fitting_data[(statistic_id, 'lag')]
            if is_crosscorr:
                phi2 = np.ones(len(fitting_data[(statistic_id, 'df')]))
                distances = fitting_data[(statistic_id, 'df')].get('distance', pd.Series([])).values
        elif is_prob:
            threshold = fitting_data[(statistic_id, 'threshold')]

        # compute appropriate statistic
        if is_mean:
            values = calculate_mean(duration_val, lamda, nu, mu_1, eta, phi)
        elif is_variance:
            values = calculate_variance(duration_val, eta, beta, lamda, nu, mu_1, mu_2, phi)
        elif is_skew:
            values = calculate_skewness(duration_val, eta, beta, lamda, nu, mu_1, mu_2, mu_3, phi)
        elif is_autocorr:
            values = calculate_autocorrelation(duration_val, lag, eta, beta, lamda, nu, mu_1, mu_2, phi)
        elif is_prob:
            # call probability dry with threshold if available (may be NaN)
            values = calculate_probability_dry(duration_val, nu, beta, eta, lamda, phi, threshold if 'threshold' in locals() else None)
        elif is_crosscorr:
            values = calculate_cross_correlation(duration_val, lag, eta, beta, lamda, nu, mu_1, mu_2, gamma, distances, phi, phi2)
        else:
            # if nothing matches, raise to avoid silent mis-assignment
            raise ValueError(f"Unknown statistic name: {name} (normalized: {name_l})")

        statistic_arrays.append(np.atleast_1d(values))

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
    duration (thresholds of 0.2 or 1.0 mm).

    """
    if h == 24:
        if threshold == 1.0:
                corr_pdry = 0.14883306000969174 + 0.9159842728203157 * uncorr_pdry 
        elif threshold == 0.2:
            corr_pdry = 0.0812432886563782 + 0.9750231398157361 * uncorr_pdry 
            

    # elif h == 1:

    #     # Burton et al. (2008) equation 10
    #     if threshold == 0.1:
    #         corr_pdry = 0.114703 + 0.884491 * uncorr_pdry

    #     # Burton et al. (2008) equation 11
    #     elif threshold == 0.2:
    #         corr_pdry = 0.239678 + 0.758837 * uncorr_pdry
        corr_pdry = max(corr_pdry, 0.0)
        corr_pdry = min(corr_pdry, 1.0)

    return corr_pdry


def calculate_probability_dry(duration, nu, beta, eta, lamda, phi, threshold=0.2):
    probability_dry = _probability_dry(duration, nu, beta, eta, lamda)
    if threshold is not None:
        probability_dry = _probability_dry_correction(duration, threshold, probability_dry)
    probability_dry = phi * 0.0 + probability_dry
    probability_dry = np.clip(probability_dry, 0.0, 1.0)
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
    Format fitted results into a consistent dataframe for NSRP point model.

    Args:
        results (dict): Optimisation results dictionary.
        all_parameter_names (list): All possible parameter names.
        parameters_to_fit (list): Parameters being fitted.
        fixed_parameters (dict): Dictionary of fixed parameters {(month,param): value}.
        unique_months (list): List of months (1-12).
        intensity_distribution (str): 'exponential', 'weibull', or 'generalised_gamma'.

    Returns:
        pd.DataFrame: Formatted dataframe with consistent column order.
    """
    dc = results.copy()

    # Insert fixed parameters if missing
    for param in all_parameter_names:
        if param not in parameters_to_fit:
            for m in unique_months:
                dc[(param, m)] = fixed_parameters.get((m, param), np.nan)

    # Convert dict -> dataframe
    df = pd.DataFrame.from_dict(dc, orient='index', columns=['value'])
    df.index = pd.MultiIndex.from_tuples(df.index, names=['field', 'month'])
    df.reset_index(inplace=True)
    df = df.pivot(index='month', columns='field', values='value')
    df.sort_index(inplace=True)
    df.reset_index(inplace=True)

    # Cast types
    type_map = {
        'month': int,
        'converged': bool,
        'iterations': int,
        'function_evaluations': int,
        'objective_function': float
    }
    for col, t in type_map.items():
        if col in df.columns:
            df[col] = df[col].astype(t)

    # Define desired order depending on distribution
    if intensity_distribution == 'exponential':
        desired_order = ['fit_stage','month','lamda','beta','nu','eta','theta',
                         'converged','objective_function','iterations','function_evaluations']
    elif intensity_distribution == 'weibull':
        desired_order = ['fit_stage','month','lamda','beta','nu','eta','theta','kappa',
                         'converged','objective_function','iterations','function_evaluations']
    elif intensity_distribution == 'generalised_gamma':
        desired_order = ['fit_stage','month','lamda','beta','nu','eta','theta','kappa_1','kappa_2',
                         'converged','objective_function','iterations','function_evaluations']
    else:
        raise ValueError(f"Unknown distribution: {intensity_distribution}")

    # Reorder, keeping only existing columns
    df = df.reindex(columns=[c for c in desired_order if c in df.columns])

    return df


## Stage-4: Fitting stage ends here and NSRP simulation stage begins ##
#############################################
# The steps in simulation of the NSRP process are:  
#     1. Simulate storms as a temporal Poisson process.
#     2. Simulate raincells.
#     3. Simulate raincell arrival times.
#     4. Simulate raincell durations.
#     5. Simulate raincell intensities.

def simulate_storms(month_lengths, simulation_length, parameters, rng):
    simulation_end_time = np.cumsum(month_lengths)[-1]

    sim_len_ext = simulation_length + 4
    month_lengths_ext = month_lengths.copy()
    for _ in range(4):
        month_lengths_ext = np.concatenate([month_lengths_ext, month_lengths[-12:]])

    lamda = np.tile(parameters['lamda'].values, sim_len_ext)
    cumulative_expected_storms = np.cumsum(lamda * month_lengths_ext)
    cumulative_month_endtimes = np.cumsum(month_lengths_ext)
    expected_number_of_storms = cumulative_expected_storms[-1]
    number_of_storms = rng.poisson(expected_number_of_storms)

    deformed_arrivals = expected_number_of_storms * np.sort(rng.uniform(size=number_of_storms))
    cumulative_expected_storms = np.insert(cumulative_expected_storms, 0, 0.0)
    cumulative_month_endtimes = np.insert(cumulative_month_endtimes, 0, 0.0)
    interpolator = scipy.interpolate.interp1d(
        cumulative_expected_storms, cumulative_month_endtimes
    )
    storm_arrival_times = interpolator(deformed_arrivals)

    storm_arrival_times = storm_arrival_times[storm_arrival_times < simulation_end_time]
    number_of_storms = storm_arrival_times.shape[0]
    storms = pd.DataFrame({
        'storm_id': np.arange(number_of_storms),
        'storm_arrival': storm_arrival_times
    })
    storms['month'] = lookup_months(month_lengths, simulation_length, storms['storm_arrival'].values)
    return storms, number_of_storms


def lookup_months(month_lengths, period_length, times):
    end_times = np.cumsum(month_lengths)
    repeated_months = np.tile(np.arange(1, 13, dtype=int), period_length)
    idx = np.digitize(times, end_times)
    return repeated_months[idx]


def simulate_raincells_point(storms, parameters, rng):
    tmp = pd.merge(storms, parameters, how='left', on='month')
    tmp.sort_values(['storm_id'], inplace=True)

    number_of_raincells = rng.poisson(tmp['nu'].values)
    storm_ids, storm_arrivals, storm_months = make_storm_arrays_by_raincell(
        number_of_raincells,
        storms['storm_id'].values,
        storms['storm_arrival'].values,
        storms['month'].values
    )
    return pd.DataFrame({
        'storm_id': storm_ids,
        'storm_arrival': storm_arrivals,
        'month': storm_months
    })


def make_storm_arrays_by_raincell(num_cells, storm_ids, storm_arrivals, storm_months):
    return (
        np.repeat(storm_ids, num_cells),
        np.repeat(storm_arrivals, num_cells),
        np.repeat(storm_months, num_cells)
    )


def merge_parameters(df, month_lengths, simulation_length, parameters):
    df['month'] = lookup_months(month_lengths, simulation_length, df['storm_arrival'].values)
    parameters_subset = parameters.drop(
        ['fit_stage', 'converged', 'objective_function',
         'iterations', 'function_evaluations'],
        axis=1, errors='ignore'
    )
    return pd.merge(df, parameters_subset, how='left', on='month')


def main_point_model_monthly(parameters,simulation_length,month_lengths,intensity_distribution,rng):
    """
    NSRP point model simulation (monthly parameters).

    Args:
        parameters (pandas.DataFrame): Parameters dataframe from fitting (must include 'month', 'lamda', 'beta', 'nu', 'eta', 'theta', etc.).
        simulation_length (int): Number of years to simulate.
        month_lengths (numpy.ndarray): Hours in each month to be simulated.
        intensity_distribution (str): Raincell intensity distribution ('exponential', 'weibull', 'generalised_gamma').
        rng (numpy.random.Generator): Random number generator.

    Steps:
        1. Simulate storms (temporal Poisson process).
        2. Simulate raincells for each storm.
        3. Simulate raincell arrival times.
        4. Simulate raincell durations.
        5. Simulate raincell intensities.
    """

    # Ensure dataframe is sorted by month
    parameters = parameters.copy()
    parameters.sort_values(by='month', inplace=True)

    # Step 1 - Simulate storms
    storms, number_of_storms = simulate_storms(month_lengths, simulation_length, parameters, rng)

    # Step 2 - Simulate raincells
    df = simulate_raincells_point(storms, parameters, rng)

    # Merge parameters into master dataframe
    df = pd.merge(df, parameters, how='left', on='month')

    # Step 3 - Raincell arrival times
    raincell_arrival_times = rng.exponential(1.0 / df['beta'])  # relative to storm origin
    df['raincell_arrival'] = df['storm_arrival'] + raincell_arrival_times

    # Step 4 - Raincell durations
    df['raincell_duration'] = rng.exponential(1.0 / df['eta'])
    df['raincell_end'] = df['raincell_arrival'] + df['raincell_duration']

    # Step 5 - Raincell intensities
    if intensity_distribution == 'exponential':
        df['raincell_intensity'] = rng.exponential(df['theta'])
    elif intensity_distribution == 'weibull':
        df['raincell_intensity'] = scipy.stats.weibull_min.rvs(
            c=df['kappa'], scale=df['theta'], random_state=rng
        )
    elif intensity_distribution == 'generalised_gamma':
        df['raincell_intensity'] = scipy.stats.gengamma.rvs(
            a=(df['kappa_1'] / df['kappa_2']), c=df['kappa_2'],
            scale=df['theta'], random_state=rng
        )

    # Clean up parameters from output (optional)
    df.drop(columns=['lamda', 'beta', 'rho', 'eta', 'gamma', 'theta', 'kappa'],
            inplace=True, errors='ignore')

    return df


def initialise_discrete_rainfall_arrays_point(n_timesteps):
    """
    Create zero-filled array for point rainfall output.

    Args:
        n_timesteps (int): Total number of timesteps in the simulation.

    Returns:
        dict: {'point': np.ndarray of shape (n_timesteps, 1)}
    """
    return {'point': np.zeros((n_timesteps, 1))}



def discretise_point(period_start_time, timestep_length,raincell_arrival_times, raincell_end_times,
                     raincell_intensities, discrete_rainfall):
    """
    Convert raincells into discrete timestep rainfall totals.

    Args:
        period_start_time (float): Simulation start time in hours.
        timestep_length (float): Length of each timestep in hours.
        raincell_arrival_times (np.ndarray): Raincell start times in hours.
        raincell_end_times (np.ndarray): Raincell end times in hours.
        raincell_intensities (np.ndarray): Raincell intensities (mm/hr).
        discrete_rainfall (np.ndarray): Array to store output (modified in place).
    """
    discrete_rainfall.fill(0.0)  # Reset to zero before filling

    for idx in range(raincell_arrival_times.shape[0]):
        # Times relative to simulation/block start
        rc_arrival_time = raincell_arrival_times[idx] - period_start_time
        rc_end_time = raincell_end_times[idx] - period_start_time
        rc_intensity = raincell_intensities[idx]

        # Timesteps covered
        rc_arrival_timestep = int(np.floor(rc_arrival_time / timestep_length))
        rc_end_timestep = int(np.floor(rc_end_time / timestep_length))

        # Distribute intensity across affected timesteps
        for timestep in range(rc_arrival_timestep, rc_end_timestep + 1):
            timestep_start_time = timestep * timestep_length
            timestep_end_time = (timestep + 1) * timestep_length
            effective_start = max(rc_arrival_time, timestep_start_time)
            effective_end = min(rc_end_time, timestep_end_time)
            timestep_coverage = effective_end - effective_start

            if timestep < discrete_rainfall.shape[0] and timestep_coverage > 0:
                discrete_rainfall[timestep, 0] += rc_intensity * timestep_coverage


def get_storm_depths_point(df):
    """
    Summarise storm total depth and duration from raincell dataframe.

    Args:
        df (pd.DataFrame): NSRP raincell output with columns:
            ['storm_id', 'storm_arrival', 'month', 'raincell_duration', 'raincell_intensity', 'raincell_end']

    Returns:
        pd.DataFrame: Storm-level statistics.
    """
    df['raincell_depth'] = df['raincell_duration'] * df['raincell_intensity']

    storm_stats = df.groupby(['storm_id']).agg({
        'storm_arrival': 'min',
        'month': 'min',
        'raincell_depth': 'sum',
        'raincell_end': 'max'
    }).reset_index()

    storm_stats.rename(columns={'raincell_depth': 'storm_depth',
                                'raincell_end': 'storm_end'}, inplace=True)
    storm_stats['storm_duration'] = storm_stats['storm_end'] - storm_stats['storm_arrival']
    storm_stats.drop(columns=['storm_end'], inplace=True)

    return storm_stats


def discretise_by_point_monthly(df_raincells, simulation_length_years, InputTimeSeries, timestep_length=24.0):
    # Input: timestep_length in hours (default 24 -> produce daily totals)
    Start = pd.to_datetime(str(int(InputTimeSeries.index.year[0])) + '-01-01')
    End = pd.to_datetime(str(int(InputTimeSeries.index.year[0]) + simulation_length_years - 1) + '-12-31')
    DatesVect = pd.date_range(start=Start, end=End, freq="D")
    monthly_df = pd.DataFrame({'Year': DatesVect.year, 'Month': DatesVect.month})
    HRCOUNT = monthly_df.groupby(['Year', 'Month']).size().reset_index(name='n_days')
    month_days = HRCOUNT['n_days'].to_numpy()

    # number of timesteps per month given timestep_length (hours)
    timesteps_per_month = (month_days * 24 / timestep_length).astype(int)  # usually days if timestep_length==24
    total_timesteps = timesteps_per_month.sum()

    # Create discrete output at the requested timestep resolution
    discrete_rainfall = np.zeros((total_timesteps, 1))

    # Loop months and use hours-based start/end for mask
    start_ts = 0
    start_hour = 0.0
    for m_idx, days_in_month in enumerate(month_days):
        hours_in_month = days_in_month * 24.0
        end_hour = start_hour + hours_in_month
        end_ts = start_ts + timesteps_per_month[m_idx]

        # mask must compare hours to hours
        mask = ((df_raincells['raincell_arrival'] < end_hour) & (df_raincells['raincell_end'] > start_hour))
        raincells_month = df_raincells.loc[mask]

        # slice the output array corresponding to this month (timesteps, 1)
        discretise_point(
            period_start_time=start_hour,
            timestep_length=timestep_length,
            raincell_arrival_times=raincells_month['raincell_arrival'].values,
            raincell_end_times=raincells_month['raincell_end'].values,
            raincell_intensities=raincells_month['raincell_intensity'].values,
            discrete_rainfall=discrete_rainfall[start_ts:end_ts]
        )

        # advance
        start_hour = end_hour
        start_ts = end_ts

    return discrete_rainfall


def initialise_hourly_array(total_hours):
    # One column for point model, all zeros initially
    return np.zeros((total_hours, 1), dtype=float)

##This marks the end of functions relevant to NSRP simulation stage. 


# Now we are dealing with HadUK dataset which has 10432 grids (5 KM grid size)
# this implies we have to take care of a few additional steps
# which include extracting data from the NetCDF files for a given grid
# Following that we begin the process of running the rainfall generator


GRIDS = pd.read_csv('/home/users/azhar199/DATA/HADUK/COMPGRID.csv')

os.chdir('/home/users/azhar199/DATA/HADUK/HADUK_RF')
FILES = glob.glob('*.nc')
RFDS = []
for i in np.arange(0,len(FILES),1):
    READ = Dataset(FILES[i])
    RF = READ.variables['rainfall'][:,GRIDS['LatIndex'][8284],GRIDS['LonIndex'][8284]]
    RFDS.append(RF)

dfb_daily = pd.DataFrame({'datetime':pd.date_range(start=pd.to_datetime('1961-01-01'), 
                                                   end=pd.to_datetime('2024-12-31'), freq="D"),
                          'value':np.concatenate(RFDS)})

dfb_daily.index = dfb_daily['datetime']
SD = {12: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 3, 9: 4, 10: 4, 11: 4}    
ALLDF = prepare_point_timeseries(dfb_daily,season_definitions=SD,completeness_threshold=0,durations=['24H','72H','1M'] ,outlier_method='trim',maximum_relative_difference=2,maximum_alterations=5)

reference_statistics = GetMonthStats(ALLDF,0.2)

unique_months = list(range(1,13))
all_parameter_names = ['lamda', 'beta', 'eta', 'nu', 'theta','kappa']
parameters_to_fit = ['lamda', 'beta', 'eta', 'nu', 'theta','kappa']
fixed_parameters = {}
parameter_bounds = {
    (m, param): bounds
    for m in range(1, 13)
    for param, bounds in {
        'lamda': (0.00001, 0.02),
        'beta': (0.02, 1),
        'eta': (0.1, 70),
        'nu': (0.1, 60),
        'theta': (0.25, 130),
        'kappa': (0.5,1.2)
    }.items()
}

def fit_month_task(month):
    return fit_by_month_point(unique_months = [month], reference_statistics = reference_statistics,
                              intensity_distribution =  'weibull',all_parameter_names = all_parameter_names, 
                              parameters_to_fit = parameters_to_fit, parameter_bounds = parameter_bounds, 
                              fixed_parameters = fixed_parameters)

if __name__ == '__main__':
    months = unique_months 
    with Pool() as pool:   
        results = pool.map(fit_month_task, months)
    parameters_df = pd.concat([res[0] for res in results], axis=0)
    fitted_stats = pd.concat([res[1] for res in results], axis=0)

parameters_df.columns = ['month','lamda','beta','nu','eta','theta','kappa','converged','objective_function','iterations','function_evaluations','fit_stage']
numeric_cols = ['lamda','beta','nu','eta','theta','kappa']
for col in numeric_cols:
    if col in parameters_df.columns:
        parameters_df[col] = pd.to_numeric(parameters_df[col], errors='coerce')
        
        
n_realizations = 300
n_years = 80
base_seed = 45
rng = np.random.default_rng(seed = base_seed)

def GetMonthLengths2(NUM):
    n_years = NUM
    Start = pd.to_datetime(str(int(dfb_daily.index.year[0]))+str('-01-01'),format='%Y-%m-%d')
    End = pd.to_datetime(str(int(dfb_daily.index.year[0]+n_years-1))+str('-12-31'),format='%Y-%m-%d')
    DatesVect = pd.date_range(start=Start,end=End,freq="D")
    monthly_df = pd.DataFrame({'Year':DatesVect.year,'Month':DatesVect.month})
    HRCOUNT = monthly_df.groupby(['Year', 'Month']).size().reset_index(name='n_hours')
    return DatesVect,HRCOUNT['n_hours'].to_numpy()

month_lengths_array = GetMonthLengths2(n_years)[1]*24
realizations = []

for i in range(n_realizations):
    rng = np.random.default_rng(seed = base_seed + i)  
    sim_df = main_point_model_monthly(parameters=parameters_df, simulation_length=n_years, month_lengths=month_lengths_array, intensity_distribution='weibull',rng=rng)
    HR = discretise_by_point_monthly(sim_df, simulation_length_years=n_years,InputTimeSeries = dfb_daily,timestep_length = 24)
    HR_df = pd.DataFrame(HR, columns=['rainfall'])
    realizations.append(HR_df)


all_realizations_df = pd.concat(realizations, ignore_index=True,axis=1)
all_realizations_df.columns = ['Realization'+'_'+str(i+1) for i in range(n_realizations)]
all_realizations_df['DateTime'] = GetMonthLengths2(n_years)[0]
all_realizations_df['Year'] = [all_realizations_df['DateTime'][i].year for i in range(all_realizations_df.shape[0])]
all_realizations_df['Month'] = [all_realizations_df['DateTime'][i].month for i in range(all_realizations_df.shape[0])]


# # Diagnosis of the results start from here #
# Firstly the statistics of observed time series of rainfall are computed #
# These statistics include - annual rainfall depths, mean monthly rainfall depths, monthly skewness, variance and dry probabilities and ACF-lag1 #

obs_annual_sum = (ALLDF['24H'].groupby(['Year'])['value'].sum().reset_index())

monthly_sum_df = (ALLDF['24H'].groupby(['Year', 'Month'])['value'].sum().reset_index())
obs_mon_mean = monthly_sum_df.groupby('Month')['value'].mean()

monthly_var_df = (ALLDF['24H'].groupby(['Year', 'Month'])['value'].var().reset_index())
obs_mon_var = monthly_var_df.groupby('Month')['value'].mean()

monthly_skew_df = (ALLDF['24H'].groupby(['Year', 'Month'])['value'].skew().reset_index())
obs_mon_skew = monthly_skew_df.groupby('Month')['value'].mean()

obs_mon_acf = reference_statistics['value'][(reference_statistics['duration']=='24H') & (reference_statistics['name'] == 'autocorrelation_lag1')]
obs_mon_pdry = reference_statistics['value'][(reference_statistics['duration']=='24H') & (reference_statistics['name'] == 'probability_dry_0.2mm')]

obs_monthly_stats = pd.DataFrame({'Month': range(1, 13),'Mean': obs_mon_mean.values,'Variance': obs_mon_var.values,'Skewness': obs_mon_skew.values,
                                'ACF_lag1': obs_mon_acf.values,'Pdry_0.2mm': obs_mon_pdry})

# Statistics mentioned in the previous cell are now computed for all the realizations #

SUBSET = all_realizations_df[all_realizations_df['Year'].isin(np.arange(ALLDF['24H']['Year'].min(),ALLDF['24H']['Year'].max()+1))]
REALIZATION_ANN_SUM = SUBSET.groupby('Year')[['Realization_' + str(j+1) for j in range(n_realizations)]].sum().reset_index()

real_cols = [f"Realization_{i+1}" for i in range(n_realizations)]
monthly_sum_real = SUBSET.groupby(['Year', 'Month'])[real_cols].sum().reset_index()
monthly_mean_real = monthly_sum_real.groupby('Month')[real_cols].mean().reset_index()

monthly_var_real = SUBSET.groupby(['Year', 'Month'])[real_cols].var().reset_index()
monthly_meanvar_real = monthly_var_real.groupby('Month')[real_cols].mean().reset_index()

monthly_skew_real = SUBSET.groupby(['Year', 'Month'])[real_cols].skew().reset_index()
monthly_meanskew_real = monthly_skew_real.groupby('Month')[real_cols].mean().reset_index()

dry_mask = (SUBSET[real_cols] < 0.2).astype(int)
dry_mask['Month'] = SUBSET['Month']
monthly_pdry_real = dry_mask.groupby('Month')[real_cols].mean().reset_index()

COL = SUBSET.columns[SUBSET.columns.str.contains('Realization_')]
daily_df_real =[]
for i in COL:
    daily_sum = (SUBSET.groupby(pd.Grouper(key='DateTime', freq='D'))[i].sum().reset_index())
    daily_df_real.append(daily_sum[daily_sum.columns[1]])
    
DAILY_DF_REAL=pd.concat(daily_df_real,axis=1)
DAILY_DF_REAL['Date'] = daily_sum[daily_sum.columns[0]]
DAILY_DF_REAL['Month'] = [DAILY_DF_REAL['Date'][i].month for i in range(0,DAILY_DF_REAL.shape[0])]

def getacf(DFNUM, MONTHNUM):
    df = pd.DataFrame({'x': DAILY_DF_REAL[DFNUM][DAILY_DF_REAL['Month'] == MONTHNUM],'x_lag': DAILY_DF_REAL[DFNUM][DAILY_DF_REAL['Month'] == MONTHNUM].shift(1)})
    df.dropna(inplace=True)
    if len(df) > 1:
       acf, pval = scipy.stats.pearsonr(df['x'], df['x_lag'])
    else:
        acf = np.nan
    return acf
    
monthly_acf_real = pd.DataFrame([[getacf(h, j) for h in COL] for j in range(1, 13)],index=range(1, 13),columns=COL)
monthly_acf_real['Month'] = range(1,13)   


# Checking the rainfall depths for various return periods #

def main(rvs):
    shape, loc, scale = gev.fit(rvs)
    return shape, loc, scale

OBS_MAX = ALLDF['24H'].groupby('Year')['value'].agg(max)
GEV_PAR = main(OBS_MAX)

def gev_return_level(mu, sigma, xi, N):
    term = (-np.log(1 - 1.0/N)) ** (-xi)
    RL = mu - (sigma/xi) * (1 - term)
    return RL

RPVAL = [gev_return_level(mu = GEV_PAR[1], sigma = GEV_PAR[2], xi = GEV_PAR[0], N = i) for i in np.arange(5,55,5)]
RPVAL_DF_OBS = pd.DataFrame({'RP':np.arange(5,55,5),'RF':RPVAL})

all_realizations_df2 = all_realizations_df[all_realizations_df['Year'].isin( np.arange(np.min(ALLDF['24H']['Year']), np.max(ALLDF['24H']['Year'])+1,1) )]

REAL_MAX = []
for i in real_cols:
    real_max = all_realizations_df2.groupby('Year')[i].agg(max)
    REAL_MAX.append(real_max)

GEV_PAR_REAL = [main(REAL_MAX[i]) for i in np.arange(0,len(REAL_MAX),1)]

RPVAL_DF_REAL = []
for b in np.arange(0,len(GEV_PAR_REAL),1):
    rpval = [gev_return_level(mu = GEV_PAR_REAL[b][1], sigma = GEV_PAR_REAL[b][2], xi = GEV_PAR_REAL[b][0], N = i) for i in np.arange(5,55,5)]
    rpval_df = pd.DataFrame({'RF':rpval})
    RPVAL_DF_REAL.append(rpval_df)

REAL_RP_DF = pd.concat(RPVAL_DF_REAL,axis=1)
REAL_RP_DF.columns = real_cols 
REAL_RP_DF['RP'] = np.arange(5,55,5)

df_long = REAL_RP_DF.melt(id_vars="RP", value_vars=real_cols, var_name="Realization", value_name="Value")


real_cols = [c for c in REALIZATION_ANN_SUM.columns if c.startswith('Realization_')]

fig, axes = plt.subplots(2, 3, figsize=(20, 10))
axes = axes.flatten()  # flatten 2x3 grid to 1D array

# List of month names
month_names = [calendar.month_abbr[i] for i in range(1, 13)]
realization_handle = Line2D([0], [0], color='darkgrey', linewidth=0.8, alpha=0.6, label='Realizations')

#  Annual plot
# axes[0].plot(REALIZATION_ANN_SUM['Year'], REALIZATION_ANN_SUM[real_cols], color='darkgrey', linewidth=0.8, alpha=0.6)
# axes[0].plot(obs_annual_sum['Year'], obs_annual_sum['value'], marker='o', color='red', linewidth=2, label='Observed')
# axes[0].set_title('Annual Sum', fontweight='bold')
# axes[0].set_xlabel('Year', fontweight='bold')
# axes[0].set_ylabel('Rainfall', fontweight='bold')
# axes[0].legend(handles=[realization_handle, axes[0].lines[-1]])
# #axes[0].set_xticklabels(obs_annual_sum['Year'], fontweight='bold')
# axes[0].set_yticklabels(axes[0].get_yticks(), fontweight='bold')

#  Monthly Mean
axes[0].plot(monthly_mean_real['Month'], monthly_mean_real[real_cols], color='darkgrey', linewidth=0.8, alpha=0.6)
axes[0].plot(obs_monthly_stats['Month'], obs_monthly_stats['Mean'], marker='o', color='red', linewidth=2)
axes[0].set_title('Monthly Mean Rainfall Depth', fontweight='bold')
axes[0].set_xlabel('Month', fontweight='bold')
axes[0].set_ylabel('Rainfall', fontweight='bold')
axes[0].set_xticks(range(1, 13))
axes[0].set_xticklabels(month_names, fontweight='bold')
axes[0].set_yticklabels(axes[0].get_yticks(), fontweight='bold')

#  Monthly Variance
axes[1].plot(monthly_meanvar_real['Month'], monthly_meanvar_real[real_cols], color='darkgrey', linewidth=0.8, alpha=0.6)
axes[1].plot(obs_monthly_stats['Month'], obs_monthly_stats['Variance'], marker='o', color='red', linewidth=2)
axes[1].set_title('Daily Variance', fontweight='bold')
axes[1].set_xlabel('Month', fontweight='bold')
axes[1].set_ylabel('Variance', fontweight='bold')
axes[1].set_xticks(range(1, 13))
axes[1].set_xticklabels(month_names, fontweight='bold')
axes[1].set_yticklabels(np.round(axes[1].get_yticks(),2), fontweight='bold')

#  Monthly Skewness
axes[2].plot(monthly_meanskew_real['Month'], monthly_meanskew_real[real_cols], color='darkgrey', linewidth=0.8, alpha=0.6)
axes[2].plot(obs_monthly_stats['Month'], obs_monthly_stats['Skewness'], marker='o', color='red', linewidth=2)
axes[2].set_title('Daily Skewness', fontweight='bold')
axes[2].set_xlabel('Month', fontweight='bold')
axes[2].set_ylabel('Skewness', fontweight='bold')
axes[2].set_xticks(range(1, 13))
axes[2].set_xticklabels(month_names, fontweight='bold')
#axes[2].set_yticklabels(axes[2].get_yticks(), fontweight='bold')

#  Monthly ACF lag-1
axes[3].plot(monthly_acf_real['Month'], monthly_acf_real[real_cols], color='darkgrey', linewidth=0.8, alpha=0.6)
axes[3].plot(obs_monthly_stats['Month'], obs_monthly_stats['ACF_lag1'], marker='o', color='red', linewidth=2)
axes[3].set_title('Lag-1 Autocorrelation', fontweight='bold')
axes[3].set_xlabel('Month', fontweight='bold')
axes[3].set_ylabel('ACF', fontweight='bold')
axes[3].set_xticks(range(1, 13))
axes[3].set_xticklabels(month_names, fontweight='bold')
axes[3].set_yticklabels(np.round(axes[3].get_yticks(),2), fontweight='bold')

#  Monthly Pdry 0.2 mm
axes[4].plot(monthly_pdry_real['Month'], monthly_pdry_real[real_cols], color='darkgrey', linewidth=0.8, alpha=0.6)
axes[4].plot(obs_monthly_stats['Month'], obs_monthly_stats['Pdry_0.2mm'], marker='o', color='red', linewidth=2)
axes[4].set_title('Dry Probability (0.2 mm)', fontweight='bold')
axes[4].set_xlabel('Month', fontweight='bold')
axes[4].set_ylabel('Probability', fontweight='bold')
axes[4].set_xticks(range(1, 13))
axes[4].set_xticklabels(month_names, fontweight='bold')
axes[4].set_yticklabels(np.round(axes[4].get_yticks(),2), fontweight='bold')


ax6 = axes[5]

sns.boxplot(data=df_long,x="RP",y="Value",ax=ax6,
    boxprops=dict(facecolor='none', edgecolor='blue'),
    medianprops=dict(color='black'),
    whiskerprops=dict(color='black'),
    capprops=dict(color='black'),
    flierprops=dict(markeredgecolor='black'),
    patch_artist=True)

# Plot observed points
for i in range(RPVAL_DF_OBS.shape[0]):
    ax6.plot(i, RPVAL_DF_OBS['RF'][i], 'ro', markersize=10)

ax6.set_title("24-Hour Rainfall Depths", weight='bold')
ax6.set_xlabel("Return Period (RP) in Year", weight='bold')
ax6.set_ylabel("Rainfall depth (mm)", weight='bold')

plt.tight_layout()
plt.show()




