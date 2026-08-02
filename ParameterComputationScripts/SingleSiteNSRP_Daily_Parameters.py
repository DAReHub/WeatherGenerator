import os
import sys
import pandas as pd
import numpy as np
from numpy import nan
import glob
from scipy.stats import skew
import scipy.stats
from scipy import stats
import calendar
from scipy.optimize import differential_evolution
from multiprocessing import Pool
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from netCDF4 import Dataset


## Stage-1: Data/time series preparation stage begins from here ##
def prepare_point_timeseries(df, season_definitions, completeness_threshold, durations, outlier_method,
        maximum_relative_difference, maximum_alterations,):
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

    # TODO: More efficient approach would be to use successive durations in aggregation
    # - e.g use 1hr to get 3hr, but then use 3hr to get 6hr, 6hr to get 12hr, etc
    # - only works if there is a neat division, otherwise need to go back to e.g. 1hr

    # Prepare order to process durations in, so that long durations can be calculated from daily rather than hourly
    # durations (as faster)
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
        # df1.drop(columns=['level_0'], inplace=True)

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

# Stage-2: We get reference statistics from this stage and 
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

        corr_pdry = max(corr_pdry, 0.0)
        corr_pdry = min(corr_pdry, 1.0)

    return corr_pdry


def calculate_probability_dry(duration, nu, beta, eta, lamda, phi, threshold=0.2):
    probability_dry = _probability_dry(duration, nu, beta, eta, lamda)
    #if threshold is not None:
        #probability_dry = _probability_dry_correction(duration, threshold, probability_dry)
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


def fit_month_task(month, reference_statistics, unique_months, all_parameter_names,
                   parameters_to_fit, parameter_bounds, fixed_parameters):
    return fit_by_month_point(
        unique_months=[month],
        reference_statistics=reference_statistics,
        intensity_distribution='weibull',
        all_parameter_names=all_parameter_names,
        parameters_to_fit=parameters_to_fit,
        parameter_bounds=parameter_bounds,
        fixed_parameters=fixed_parameters
    )




GRIDS = pd.read_csv('/home/users/azhar199/DATA/NEWDATA/GRIDDED_WG_QC/HOURLY_DAILY/HUKMETA2.csv')
RF_MEAN_CF = pd.read_csv(f'/home/users/azhar199/DATA/NEWDATA/GRIDDED_WG/UKCP_CF/RFMEAN/RF_MEAN_ChangeFactors_HUK_M01.csv')
GRIDS2 = GRIDS.merge(RF_MEAN_CF[['northing','easting']].drop_duplicates()).reset_index(drop=True)

def GetRefs_REGION(NUM):
    region = GRIDS2.iloc[NUM]['Region']
    os.chdir('/home/users/azhar199/DATA/HADUK/HADUK_RF')
    FILES = glob.glob('*.nc')
    FILES = FILES[480:720] 
    RFDS = []
    for i in np.arange(0,len(FILES),1):
        READ = Dataset(FILES[i])
        RF = READ.variables['rainfall'][:,GRIDS2['HUK_y_index'][NUM],GRIDS2['HUK_x_index'][NUM]]
        RFDS.append(RF)
    
    dfb_daily = pd.DataFrame({'datetime':pd.date_range(start=pd.to_datetime('2001-01-01'), end=pd.to_datetime('2020-12-31'), freq="D"),'value':np.concatenate(RFDS)})
    dfb_daily.index = dfb_daily['datetime']
    SD = {12: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 3, 9: 4, 10: 4, 11: 4}    
    ALLDF = prepare_point_timeseries(dfb_daily,season_definitions=SD,completeness_threshold=0,durations=['24H','72H','1M'] ,outlier_method='trim',maximum_relative_difference=2,maximum_alterations=5)
    
    refdaily = GetMonthStats(ALLDF,1)
    
    refdaily.loc[(refdaily['name']=='mean') & (refdaily['duration']=='24H'),'statistic_id'] = 4
    refdaily.loc[(refdaily['name']=='variance') & (refdaily['duration']=='24H'),'statistic_id'] = 5
    refdaily.loc[(refdaily['name']=='skewness') & (refdaily['duration']=='24H'),'statistic_id'] = 6
    refdaily.loc[(refdaily['name'].str.contains('probability_dry')) &(refdaily['duration']=='24H'),'statistic_id'] = 7
    refdaily.loc[refdaily['name']=='autocorrelation_lag1','statistic_id'] = 8
    refdaily.loc[(refdaily['duration']=='72H') &(refdaily['name']=='variance'),'statistic_id'] = 9
    refdaily = refdaily[refdaily['duration']!='1M']
    refdaily['easting'] = GRIDS2.iloc[NUM]['easting']
    refdaily['northing'] = GRIDS2.iloc[NUM]['northing']

    
    MEM = ['M01','M04','M05','M06','M07','M08','M09','M10','M11','M12','M13','M15','M23','M25','M27','M29']
    CFLIST = []
    for i in MEM:
        RF_MEAN_CF = pd.read_csv(f'/home/users/azhar199/DATA/NEWDATA/GRIDDED_WG/UKCP_CF/RFMEAN/RF_MEAN_ChangeFactors_HUK_{i}.csv')
        RF_MEAN_CF['VAR'] = 'RF'
        PDD_CF_24H = pd.read_csv(f'/home/users/azhar199/DATA/NEWDATA/GRIDDED_WG/UKCP_CF/PDD/H24_1MM/PDD_1MM_ChangeFactors_H24_{i}.csv')
        PDD_CF_24H['VAR'] = 'PDD_24H'
        ACF_CF = pd.read_csv(f'/home/users/azhar199/DATA/NEWDATA/GRIDDED_WG/UKCP_CF/ACF/ACF_ChangeFactors_HUK_{i}.csv')
        ACF_CF['VAR'] = 'ACF'
        VAR24H = pd.read_csv(f'/home/users/azhar199/DATA/NEWDATA/GRIDDED_WG/UKCP_CF/RFVAR_24H/RFVAR_24H_HUK_{i}.csv')
        VAR72H = pd.read_csv(f'/home/users/azhar199/DATA/NEWDATA/GRIDDED_WG/UKCP_CF/RFVAR_72H/RFVAR_72H_HUK_{i}.csv')
        SKEW24H = pd.read_csv(f'/home/users/azhar199/DATA/NEWDATA/GRIDDED_WG/UKCP_CF/RFSKEW_24H/RFSKEW_24H_HUK_{i}.csv')
        VAR24H = VAR24H[['northing','easting','month','VAR_CF_2021_2040','VAR_CF_2041_2060','VAR_CF_2061_2080']]
        VAR72H = VAR72H[['northing','easting','month','VAR_CF_2021_2040','VAR_CF_2041_2060','VAR_CF_2061_2080']]
        SKEW24H = SKEW24H[['northing','easting','month','SKEW_CF_2021_2040','SKEW_CF_2041_2060','SKEW_CF_2061_2080']]
        COLNAMES =  ['northing','easting','month','ChangeFactor_2021_2040','ChangeFactor_2041_2060','ChangeFactor_2061_2080']
        VAR24H.columns = COLNAMES
        VAR72H.columns = COLNAMES
        SKEW24H.columns = COLNAMES
        VAR24H['VAR'] = 'VAR24H'
        VAR72H['VAR'] = 'VAR72H'
        SKEW24H['VAR'] = 'SKEW24H'
        CF_DF = pd.concat([RF_MEAN_CF,PDD_CF_24H,ACF_CF,VAR24H,VAR72H,SKEW24H],axis=0).reset_index(drop=True)
        CF_FIN = CF_DF[(CF_DF['easting'] == GRIDS2[['easting','northing']].iloc[NUM][0]) & (CF_DF['northing'] == GRIDS2[['easting','northing']].iloc[NUM][1])].reset_index(drop=True)
        CFLIST.append(CF_FIN)
    
    
    MEMREFS = []
    for d in np.arange(0,len(CFLIST),1):
        reference_statistics = refdaily.copy()
        CFCOL = 'ChangeFactor_2061_2080'
        RF_MEAN = CFLIST[d][CFLIST[d]['VAR'] == 'RF']
        rf_mean_map = RF_MEAN.set_index('month')[CFCOL]
        mask = reference_statistics['name'].isin(['mean'])
        reference_statistics.loc[mask, 'value'] *= (reference_statistics.loc[mask, 'month'].map(rf_mean_map))
        
        mask2 = reference_statistics['name'] == 'autocorrelation_lag1'
        ACF_MEAN = CFLIST[d][CFLIST[d]['VAR'] == 'ACF']
        acf_map = ACF_MEAN.set_index('month')[CFCOL]
        r_obs = reference_statistics.loc[mask2, 'value']
        W_obs = (1 + r_obs) / (1 - r_obs)
        alpha = reference_statistics.loc[mask2, 'month'].map(acf_map)
        W_fut = alpha * W_obs
        r_fut = (W_fut - 1) / (W_fut + 1)
        reference_statistics.loc[mask2, 'value'] = r_fut
        
        mask3 = ((reference_statistics['name'] == 'probability_dry_1mm') & (reference_statistics['duration'] == '24H'))
        PDD_MEAN = CFLIST[d][CFLIST[d]['VAR'] == 'PDD_24H']
        CF_PDD = PDD_MEAN.set_index('month')[CFCOL]
        pdd_obs = reference_statistics.loc[mask3, 'value']
        beta = reference_statistics.loc[mask3, 'month'].map(CF_PDD)
        odds_obs = pdd_obs / (1 - pdd_obs)
        odds_fut = beta * odds_obs
        pdd_fut = odds_fut / (1 + odds_fut)
        reference_statistics.loc[mask3, 'value'] = pdd_fut

        VAR24H = CFLIST[d][CFLIST[d]['VAR'] == 'VAR24H']
        var24h_map = VAR24H.set_index('month')[CFCOL]
        mask6 = ((reference_statistics['name'] == 'variance') & (reference_statistics['duration'] == '24H'))
        reference_statistics.loc[mask6, 'value'] *= (reference_statistics.loc[mask6, 'month'].map(var24h_map))
    
        VAR72H = CFLIST[d][CFLIST[d]['VAR'] == 'VAR72H']
        var72h_map = VAR72H.set_index('month')[CFCOL]
        mask7 = ((reference_statistics['name'] == 'variance') & (reference_statistics['duration'] == '72H'))
        reference_statistics.loc[mask7, 'value'] *= (reference_statistics.loc[mask7, 'month'].map(var72h_map))

        SKEW24H = CFLIST[d][CFLIST[d]['VAR'] == 'SKEW24H']
        skew24h_map = SKEW24H.set_index('month')[CFCOL]
        mask9 = ((reference_statistics['name'] == 'skewness') & (reference_statistics['duration'] == '24H'))
        reference_statistics.loc[mask9, 'value'] *= (reference_statistics.loc[mask9, 'month'].map(skew24h_map))
        MEMREFS.append(reference_statistics)
    
    for i in np.arange(0,len(MEMREFS),1):
            MEMREFS[i]['MEM'] = MEM[i]

    return MEMREFS


def GetAllUKCP(GRDNUM):
    PARLIST = []
    for i in np.arange(0,16,1):
        reference_statistics = GetRefs_REGION(GRDNUM)[i]
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
                'kappa': (0.5,1.5)
            }.items()
        }

        results = [fit_month_task(month,reference_statistics=reference_statistics, unique_months=unique_months, all_parameter_names=all_parameter_names,
                                  parameters_to_fit=parameters_to_fit, parameter_bounds=parameter_bounds, fixed_parameters=fixed_parameters) for month in unique_months]
        
        parameters_df = pd.concat([res[0] for res in results], axis=0)
        parameters_df.index=np.arange(0,parameters_df.shape[0],1)  
        parameters_df.columns = ['month','lamda','beta','nu','eta','theta','kappa','converged','objective_function','iterations','function_evaluations','fit_stage']
        numeric_cols = ['lamda','beta','nu','eta','theta','kappa']
        for col in numeric_cols:
            if col in parameters_df.columns:
                parameters_df[col] = pd.to_numeric(parameters_df[col], errors='coerce')
        
        parameters_df = parameters_df.reset_index(drop=True)
        parameters_df['easting'] = reference_statistics['easting'].iloc[0]
        parameters_df['northing'] = reference_statistics['northing'].iloc[0]
        parameters_df['member'] = reference_statistics['MEM'].iloc[0]
        PARLIST.append(parameters_df)
    
    PARDF = pd.concat(PARLIST,axis=0).reset_index(drop=True)
    return PARDF

if __name__ == "__main__":
    GRDNUM = int(sys.argv[1])
    outdir = "/home/users/azhar199/DATA/NEWDATA/GRIDDED_WG_QC/HOURLY_DAILY/PARAM5KM/2061_2080_DAILY_NEW"
    os.makedirs(outdir,exist_ok=True)
    RPDF = GetAllUKCP(GRDNUM)
    ref_path = os.path.join(outdir, f"GRID_{GRDNUM}_PARAMS_2061_2080.parquet")
    RPDF.to_parquet(ref_path,index=False)
