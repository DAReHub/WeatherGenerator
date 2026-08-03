import os
import pandas as pd
import numpy as np
from numpy import nan
import glob
import numba
import datetime
from scipy.stats import skew
import scipy.stats
from scipy import stats
import statsmodels.api as sm
import calendar
from scipy.optimize import differential_evolution
from multiprocessing import Pool
import warnings
import itertools
warnings.simplefilter(action='ignore', category=FutureWarning)
from scipy.stats import genextreme as gev
from netCDF4 import Dataset
from scipy.stats import gumbel_r
from scipy.interpolate import interp1d
from scipy.stats import gumbel_r
from convertbng.util import convert_bng, convert_lonlat
from concurrent.futures import ProcessPoolExecutor, as_completed



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


def discretise_by_point_monthly(df_raincells, simulation_length_years, STARTYR, timestep_length=24.0):
    # Input: timestep_length in hours (default 24 -> produce daily totals)
    Start = pd.to_datetime(str(STARTYR)+str('-01-01 00:00:00'),format='%Y-%m-%d %H:%M:%S')
    End = pd.to_datetime(str(STARTYR+simulation_length_years-1)+str('-12-31 23:00:00'),format='%Y-%m-%d %H:%M:%S')
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




def prepare_weather_series(input_timeseries,input_variables,calculation_period,
    completeness_threshold,wet_threshold,season_length,point_id=1):
    
    # Read data
    #df = pd.read_csv(input_timeseries, index_col=0, parse_dates=True, infer_datetime_format=True)
    df = input_timeseries
    df.columns = [column.lower() for column in df.columns]

    # Assign month or half-month identifiers
    if season_length == 'half-month':
        df['season'] = identify_half_months(df.index)
    elif season_length == 'month':
        df['season'] = df.index.month

    # Subset on calculation period
    if calculation_period is not None:
        df = df.loc[(df.index.year >= calculation_period[0]) & (df.index.year <= calculation_period[1])]
    
    period_length = (datetime.datetime(calculation_period[1], 12, 31) - datetime.datetime(calculation_period[0], 1, 1))
    period_length = period_length.days + 1
    
    # Check enough data to continue
    if df.shape[0] >= 365:
        # Add wet day indicator
        df['wet_day'] = np.where(np.isfinite(df['prcp']) & (df['prcp'] >= wet_threshold), 1, 0)
        df['wet_day'] = np.where(~np.isfinite(df['prcp']), np.nan, df['wet_day'])

        # Compute derived temperature variables if present
        if 'temp_min' in df.columns and 'temp_max' in df.columns:
            df['temp_avg'] = (df['temp_min'] + df['temp_max']) / 2.0
            df['dtr'] = df['temp_max'] - df['temp_min']

        # Identify completeness by variable
        completeness = {}
        for variable in input_variables:
            if variable in df.columns:
                if df.shape[0] > 0:
                    if (variable in ['temp_avg', 'dtr']) and ('prcp' in df.columns):
                        completeness[variable] = (
                            np.sum(np.isfinite(df['prcp']) & np.isfinite(df[variable])) / period_length * 100
                        )
                    elif ('prcp' in df.columns) and ('temp_avg' in df.columns):
                        completeness[variable] = (
                            np.sum(
                                np.isfinite(df['prcp']) &
                                np.isfinite(df['temp_avg']) &
                                np.isfinite(df[variable])
                            ) / period_length * 100
                        )
                    else:
                        completeness[variable] = 0.0
                    completeness[variable] = min(completeness[variable], 100.0)
                    if completeness[variable] < completeness_threshold:
                        df.drop(columns=[variable], inplace=True)
                else:
                    completeness[variable] = 0.0
            else:
                completeness[variable] = 0.0

        # Need at least one variable to be sufficiently complete
        if max(completeness.values()) >= completeness_threshold:
            if 'datetime' not in df.columns:
                df.reset_index(inplace=True)

            # Reshape to long format
            df = pd.melt(df, id_vars=['datetime', 'season', 'prcp', 'wet_day'])

            # Filter relevant variables
            df = df.loc[~df['variable'].isin(['temp_mean', 'temp_min', 'temp_max', 'rel_hum'])]

            # Transition states
            df['wet_day_lag1'] = df['wet_day'].shift(1)
            df['wet_day_lag2'] = df['wet_day'].shift(2)
            df['transition'] = 'NA'
            df['transition'] = np.where((df['wet_day_lag1'] == 1) & (df['wet_day'] == 1), 'WW', df['transition'])
            df['transition'] = np.where((df['wet_day_lag1'] == 0) & (df['wet_day'] == 1), 'DW', df['transition'])
            df['transition'] = np.where((df['wet_day_lag1'] == 1) & (df['wet_day'] == 0), 'WD', df['transition'])
            df['transition'] = np.where((df['wet_day_lag1'] == 0) & (df['wet_day'] == 0), 'DD', df['transition'])
            df['transition'] = np.where(
                (df['wet_day_lag2'] == 0) & (df['wet_day_lag1'] == 0) & (df['wet_day'] == 0), 'DDD', df['transition']
            )
            df.drop(columns=['wet_day', 'wet_day_lag1', 'wet_day_lag2'], inplace=True)

            # Move precipitation into variable form
            tmp1 = df.loc[
                df['variable'] == df['variable'].unique()[0], ['datetime', 'season', 'transition', 'prcp']
            ].copy()
            tmp1.rename(columns={'prcp': 'value'}, inplace=True)
            tmp1['variable'] = 'prcp'
            df.drop(columns=['prcp'], inplace=True)
            df = pd.concat([df, tmp1])

            # Compute seasonal stats and z-scores
            df1 = df.loc[df['transition'] != 'NA']
            df1 = df1.groupby(['variable', 'season'])['value'].agg(['mean', 'std']).reset_index()
            df = pd.merge(df, df1, on=['variable', 'season'])
            df['z_score'] = (df['value'] - df['mean']) / df['std']
            df.drop(columns=['mean', 'std'], inplace=True)

            # Add point ID
            df['point_id'] = point_id
            df1['point_id'] = point_id

        else:
            df, completeness, df1 = None, None, None

    else:
        df, completeness, df1 = None, None, None

    return df, df1, completeness


def identify_half_months(date_series):
    half_months = np.zeros(date_series.shape[0], dtype=int)
    current_half_month = 1
    for month in range(1, 13):
        half_months[(date_series.month == month) & (date_series.day <= 15)] = current_half_month
        current_half_month += 1
        half_months[(date_series.month == month) & (date_series.day > 15)] = current_half_month
        current_half_month += 1
    return half_months

def do_regression(TRANSFORMED_SERIES,input_variables):
    # df2 is wide df - not in self.data_series
    df2 = TRANSFORMED_SERIES[1]  # refactor variable name ultimately

    # Set up factors to loop
    pool_ids = df2['pool_id'].unique().tolist()
    transitions = df2['transition'].unique().tolist()
    if 'NA' in transitions:
        transitions.remove('NA')
    seasons = np.unique(df2['season'])
    variables = input_variables.copy()
    outputs = [];parameters = {};residuals = {};r2 = {};standard_errors = {}
    predictors = {
        ('temp_avg', 'DDD'): ['temp_avg_lag1'],
        ('temp_avg', 'DD'): ['temp_avg_lag1'],
        ('temp_avg', 'DW'): ['temp_avg_lag1', 'prcp'],
        ('temp_avg', 'WD'): ['temp_avg_lag1', 'prcp_lag1'],
        ('temp_avg', 'WW'): ['temp_avg_lag1'],
        ('dtr', 'DDD'): ['dtr_lag1'],
        ('dtr', 'DD'): ['dtr_lag1'],
        ('dtr', 'DW'): ['dtr_lag1', 'prcp'],
        ('dtr', 'WD'): ['dtr_lag1', 'prcp_lag1'],
        ('dtr', 'WW'): ['dtr_lag1'],
        ('vap_press', 'DDD'): ['vap_press_lag1', 'temp_avg', 'dtr'],  # 'prcp',
        ('vap_press', 'DD'): ['vap_press_lag1', 'temp_avg', 'dtr'],  # 'prcp',
        ('vap_press', 'DW'): ['vap_press_lag1', 'prcp', 'temp_avg', 'dtr'],
        ('vap_press', 'WD'): ['vap_press_lag1', 'temp_avg', 'dtr'],  # 'prcp',
        ('vap_press', 'WW'): ['vap_press_lag1', 'prcp', 'temp_avg', 'dtr'],
        ('wind_speed', 'DDD'): ['wind_speed_lag1', 'temp_avg', 'dtr'],  # 'prcp',
        ('wind_speed', 'DD'): ['wind_speed_lag1', 'temp_avg', 'dtr'],  # 'prcp',
        ('wind_speed', 'DW'): ['wind_speed_lag1', 'prcp', 'temp_avg', 'dtr'],
        ('wind_speed', 'WD'): ['wind_speed_lag1', 'temp_avg', 'dtr'],  # 'prcp',
        ('wind_speed', 'WW'): ['wind_speed_lag1', 'prcp', 'temp_avg', 'dtr'],
        ('sun_dur', 'DDD'): ['sun_dur_lag1', 'temp_avg', 'dtr'],  # 'prcp',
        ('sun_dur', 'DD'): ['sun_dur_lag1', 'temp_avg', 'dtr'],  # 'prcp',
        ('sun_dur', 'DW'): ['sun_dur_lag1', 'prcp', 'temp_avg', 'dtr'],
        ('sun_dur', 'WD'): ['sun_dur_lag1', 'temp_avg', 'dtr'],  # 'prcp',
        ('sun_dur', 'WW'): ['sun_dur_lag1', 'prcp', 'temp_avg', 'dtr'],
    }
    for pool_id, season, transition, variable in itertools.product(pool_ids, seasons, transitions, variables):

        # Subset on relevant finite values - successively for dependent and then each independent variable
        df2a = df2.loc[
            (df2['pool_id'] == pool_id) & (df2['season'] == season) & (df2['transition'] == transition)
            & (np.isfinite(df2[variable]))]
        for predictor in predictors[(variable, transition)]:
            df2a = df2a.loc[np.isfinite(df2a[predictor])]

        # Populate array for regression
        n_times = df2a.shape[0]
        n_predictors = len(predictors[(variable, transition)])
        X = np.zeros((n_times, n_predictors))
        col_idx = 0
        for predictor in predictors[(variable, transition)]:
            X[:,col_idx] = df2a[predictor].values
            col_idx += 1

        # Set a minimum number of days for performing regression - as user option?
        if X.shape[0] >= 10:

            # Need regression parameters, r-squared and residuals for spatial correlation
            X = sm.add_constant(X)  # adds column of ones - required for intercept to be estimated
            model = sm.OLS(df2a[variable].values, X)
            results = model.fit()
            parameters[(pool_id, season, variable, transition)] = results.params
            df2b = df2a[['datetime', 'pool_id', variable]].copy()
            df2b['residual'] = results.resid
            residuals[(pool_id, season, variable, transition)] = df2b

            # Calculate r2 by point (not pool)
            df2b['fitted'] = results.fittedvalues
            df2c = df2b.groupby('pool_id')[[variable, 'fitted']].corr().unstack().iloc[:, 1]  # series
            df2c = df2c.to_frame('r')
            df2c['r2'] = df2c['r'] ** 2
            df2c.reset_index(inplace=True)
            for _, row in df2c.iterrows():
                r2[(row['pool_id'], season, variable, transition)] = row['r2']

            df2d = df2b.groupby('pool_id')['residual'].std()
            df2d = df2d.to_frame('residual')
            df2d.reset_index(inplace=True)
            for _, row in df2d.iterrows():
                standard_errors[(row['pool_id'], season, variable, transition)] = row['residual']
            outputs.append((pool_id, season, transition, variable, df2a, df2b, df2c, df2d))

        else:
            print(season, transition, variable)
            
    return outputs,parameters, residuals, r2, standard_errors


## The data prepearation, preprocessing and fitting stages for the weather variables ends here     
## The functions below are responsible for simulation of weather variables
## The wg.simulate of RWGEN simulates the weather variables and the functions in the background are below


def aggregate_rainfall(x, n_points, window_size):
    """
    Aggregate hourly rainfall to daily totals (sum over each 24-hour period).
    Handles both DataFrame and NumPy array inputs.
    """
    # Convert DataFrame to numeric numpy array if needed
    if isinstance(x, pd.DataFrame):
        # keep only numeric columns
        x = x.select_dtypes(include=[np.number]).to_numpy()

    n_days = x.shape[0] // window_size
    y = np.zeros((n_days, n_points))
    i = 0
    for d in range(n_days):
        y[d, :] = np.sum(x[i:i + window_size, :], axis=0)
        i += window_size
    return y


def day_of_year(year, month):
    if check_if_leap_year(year):
        pseudo_year = 2000
    else:
        pseudo_year = 2001

    doy_list = []
    d = datetime.datetime(pseudo_year, month, 1, 0)
    while (d.year == pseudo_year) and (d.month == month):
        doy_list.append(d.timetuple().tm_yday)
        d += datetime.timedelta(days=1)
    doy_array = np.asarray(doy_list)

    return doy_array

def check_if_leap_year(year):
    if year % 4 == 0:
        if year % 100 == 0:
            if year % 400 == 0:
                leap_year = True
            else:
                leap_year = False
        else:
            leap_year = True
    else:
        leap_year = False
    return leap_year



@numba.jit(nopython=True)
def regressions(n_days, season_length, month, variable, sn_sample, ri, transition_key, z_scores, output_type, transitions,
        parameters, pool_id, predictors, interpolated_parameters, residuals,
):
    for day in range(1, n_days + 1):

        # Identify season based on month (argument) and day of month if using half-months
        if season_length == 'month':
            season = month
        elif season_length == 'half-month':
            if day <= 15:  # TODO: Check looping days here and getting half-months correct (hardcoded)
                season = (month - 1) * 2 + 1
            else:
                season = (month - 1) * 2 + 2

        # Prepare (simulate) standard normal residual/error term
        # residuals = self.rng.standard_normal(1)[0]  # !221209
        residuals[:] = sn_sample[ri]  # !221209

        # Prediction of standardised anomalies
        # - day loop starts with one
        # - only 31 days in transitions array (i.e. current month)
        # - first value to store is in position 2
        for transition_id, transition_name in transition_key.items():

            # Intercept
            z_scores[(output_type, variable)][day + 1, :] = np.where(
                transitions[day - 1, :] == transition_id,
                parameters[(pool_id, season, variable, transition_name)][0],
                z_scores[(output_type, variable)][day + 1, :]
            )

            # Multiplicative terms
            i = 1
            for predictor in predictors[(variable, transition_name)]:
                # either (predictor, lag) tuples in self.predictors or parse here
                # e.g. something like predictor, lag = predictor_variable.split('_')
                if predictor.endswith('_lag1'):  # TODO: Can this parsing be replaced with a lookup?
                    predictor_variable = predictor.replace('_lag1', '')
                    lag = 1
                else:
                    predictor_variable = predictor
                    lag = 0
                if predictor_variable != 'na':
                    z_scores[(output_type, variable)][day + 1, :] += np.where(
                        transitions[day - 1, :] == transition_id,
                        parameters[(pool_id, season, variable, transition_name)][i]
                        * z_scores[(output_type, predictor_variable)][day + 1 - lag, :],
                        0.0
                    )
                i += 1

            # Scale residual/error term by standard error
            residuals *= np.where(
                transitions[day - 1, :] == transition_id,
                interpolated_parameters[('se', output_type, variable, season, transition_name)],
                1.0
            )

        # Add residual/error component
        z_scores[(output_type, variable)][day + 1, :] += residuals

        # Increment counter (index place) for standard normal sample
        ri += 1

    return z_scores[(output_type, variable)][2:, :], ri


def calculate_pet2(year, month,output_types,values,discretisation_metadata,n_points,latitude,wind_height):

    doy = day_of_year(year, month)
    n_days = doy.shape[0]
    
    for output_type in output_types:

        # Derive minimum and maximum temperatures and convert temperatures from [C] to [K]
        tmax = (values[(output_type, 'temp_avg')][2:2+n_days, :] + 0.5 * values[(output_type, 'dtr')][2:2+n_days, :])
        tmin = (values[(output_type, 'temp_avg')][2:2+n_days, :] - 0.5 * values[(output_type, 'dtr')][2:2+n_days, :])
        tmax += 273.15
        tmin += 273.15
        tavg = values[(output_type, 'temp_avg')][2:2+n_days, :] + 273.15
        elev = discretisation_metadata[(output_type, 'z')]
        pres = 101.3 * (((293.0 - (0.0065 * elev)) / 293.0) ** 5.26)
        avp = values[(output_type, 'vap_press')][2:2+n_days, :]
        avp = np.maximum(avp, 0.000001)
        
        svp_tmin = 0.6108 * np.exp((17.27 * (tmin - 273.15)) / ((tmin - 273.15) + 237.3))
        svp_tmax = 0.6108 * np.exp((17.27 * (tmax - 273.15)) / ((tmax - 273.15) + 237.3))
        svp = 0.5 * (svp_tmin + svp_tmax)

        dsvp = ((4098.0 * (0.6108 * np.exp((17.27 * (tavg - 273.15)) / ((tavg - 273.15) + 237.3)))) / (((tavg - 273.15) + 237.3) ** 2))
        dr = 1.0 + (0.033 * np.cos(((2.0 * np.pi) / 365.0) * doy)) 
        dec = 0.409 * np.sin((((2.0 * np.pi) / 365.0) * doy) - 1.39) 
        dr = dr[:, None]  # adds a second dimension
        dec = dec[:, None]
        lat = np.zeros((doy.shape[0], n_points[output_type]))  
        lat.fill(latitude)
        omega = np.arccos(-1 * np.tan(lat) * np.tan(dec))
        ra = (((24.0 * 60.0) / np.pi) * 0.0820 * dr * ((omega * np.sin(lat) * np.sin(dec)) + (np.cos(lat) * np.cos(dec) * np.sin(omega)))) 
        N = 24 / np.pi * omega

        SunshineHours = values[(output_type, 'sun_dur')][2:2+n_days, :]
        rs = (0.25 + 0.5 * (np.minimum(SunshineHours, N) / N)) * ra  
        rns = 0.77 * rs  
        rso = (0.75 + (0.00002 * discretisation_metadata[(output_type, 'z')])) * ra
        rnl = ((4.903 * 10 ** -9) * ((tmin ** 4.0 + tmax ** 4.0) / 2.0) * (0.34 - (0.14 * (avp ** 0.50))) * ((1.35 * (rs / rso)) - 0.35))
        netrad = rns - rnl

        Windspeed = values[(output_type, 'wind_speed')][2:2+n_days, :]

        # - adjustment from 10 to 2m is required for MIDAS data
        if wind_height != 2.0:
            ws2 = Windspeed * (4.87 / (np.log((67.8 * float(wind_height)) - 5.42)))
        else:
            ws2 = Windspeed

        # Assume soil heat flux of zero at daily timestep (FAO56 equation 42)
        shf = 0.0
        psy = 0.000665 * pres
        # Calculate ET0 [mm day-1]
        et0 = (((0.408 * dsvp * (netrad - shf)) + (psy * (900.0 / tavg) * ws2 * (svp - avp))) / (dsvp + (psy * (1.0 + (0.34 * ws2)))))
        return et0   

def interpolated_parameters_point(transitions,simulation_variables,seasons,raw_statistics,r2,standard_errors,parameters):
    
    tmp = []  # temporary list for looking at r2  # TODO: Remove
    tmp_pars = []
    interpolated_parameters = {}
    for season, variable, in itertools.product(seasons,simulation_variables):
        # raw statistics
        for statistic in ['mean', 'std']:
            interpolated_parameters[('raw_statistics', 'point', variable, season, statistic)] = (raw_statistics.loc[(raw_statistics['variable'] == variable) & (raw_statistics['season'] == season),statistic].values)
        # r2 and se
        point_id = 1
        for transition in transitions:
            key = (point_id, season, variable, transition)
            if key in r2.keys():
                interpolated_parameters[('r2', 'point', variable, season, transition)] = r2[key]
                interpolated_parameters[('se', 'point', variable, season, transition)] = (standard_errors[key])
                tmp.append([variable, season, transition,r2[key]])
            _ = [variable, season, transition]
            key = (1, season, variable, transition)
            if key in parameters.keys():
                _.extend(parameters[(1, season, variable, transition)].tolist())
                tmp_pars.append(_)
                
    return interpolated_parameters

def simulate_daily_weather_point(RAINFALL_REALIZATIONS, year, month, n_realizations, LATITUDE_DEGREES, LONGITUDE_DEGREES, point_elevation, predictors, input_variables, transitions, seasons, raw_statistics, r2, standard_errors, parameters, timestep, output_types,
                                 n_points, transformations, transformed_statistics_dict, output_variables, wet_threshold, season_length, wind_height, offset_df, base_seed, realization_counter):
    
    RF2 = RAINFALL_REALIZATIONS[(RAINFALL_REALIZATIONS['Year']==year) & (RAINFALL_REALIZATIONS['Month']==month)]
    n_timesteps = RF2.shape[0] 
    RAINFALL = []
    for r in range(1,n_realizations+1):
        rain_input = {'point': RF2[[f'Realization_{r}']].rename(columns={f'Realization_{r}': 'rainfall'})}
        RAINFALL.append(rain_input)
    
    season = month
    transition_key_nb = numba.typed.Dict.empty(numba.types.int64, numba.types.string)  # numba.types.unicode_type
    z_scores_nb = numba.typed.Dict.empty(numba.types.UniTuple(numba.types.string, 2), numba.float64[:,:])
    parameters_nb = numba.typed.Dict.empty(numba.types.Tuple([numba.types.int64, numba.types.int64, numba.types.string, numba.types.string]),numba.types.float64[:])
    predictors_nb = numba.typed.Dict.empty(numba.types.UniTuple(numba.types.string, 2), numba.types.UniTuple(numba.types.string, 4))
    interpolated_parameters_nb = numba.typed.Dict.empty(numba.types.Tuple([numba.types.string, numba.types.string, numba.types.string, numba.types.int64,numba.types.string]),numba.types.float64[:])
    transition_key = {1: 'DDD',2: 'DD',3: 'DW',4: 'WD',5: 'WW'}
    
    discretisation_metadata = {('point', 'x'): np.array([LONGITUDE_DEGREES]), 
                               ('point', 'y'): np.array([LATITUDE_DEGREES*np.pi/180]),
                               ('point', 'z'): np.array([point_elevation])}
    
    pool_id = 1 # this is a default for single-site simulations

    for k, v in transition_key.items():
        transition_key_nb[k] = v
            
    for k, v in parameters.items():
        parameters_nb[k] = v
    
    for k, vs in predictors.items():
        tmp = []
        i = 0
        for v in vs:
            tmp.append(v)
            i += 1
        while i < 4:
              tmp.append('na')
              i += 1
        predictors_nb[k] = tuple(tmp)

    simulation_variables = input_variables.copy()
    simulation_variables.append('prcp')
    interpolated_parameters = interpolated_parameters_point(transitions = transitions, 
                                                            simulation_variables = simulation_variables,
                                                            seasons = seasons, 
                                                            raw_statistics = raw_statistics, 
                                                            r2 = r2, 
                                                            standard_errors = standard_errors, 
                                                            parameters = parameters)
    rows = []
    for var in input_variables:
        mean_val = interpolated_parameters[('raw_statistics','point', var, season, 'mean')][0]
        sd_val = interpolated_parameters[('raw_statistics','point', var, season, 'std')][0]
        rows.append({'variable': var, 'mean': mean_val, 'sd': sd_val})
    
    IDP_DF = pd.DataFrame(rows)
    for k, v in interpolated_parameters.items():
        if isinstance(v, float):
            v_ = np.asarray([v])
        else:
            v_ = v
        interpolated_parameters_nb[k] = v_

    n_days = int(n_timesteps / (24 / timestep))
    _n = n_days * len(input_variables) * len(output_types)
    rng = np.random.default_rng(seed = base_seed)
    sn_sample = rng.standard_normal(_n)  # standard normal sample
    ri = 0  # counter for residual - for indexing sn_sample (increment after each day+variable combination)

    # daily weather values for current month are stored in values and lag_values will be lag of daily values, 
    #as that is what underpins regressions
    z_scores = {}; values = {}; lag_z_scores = {}; lag_values = {}
    offset_season = offset_df[offset_df['season'] == season].reset_index(drop=True)
    LAMDF = pd.DataFrame({'variable':['temp_avg', 'dtr'],'lamda':[transformations[(pool_id, var, season, 'lamda')] for var in ['temp_avg', 'dtr']]})
    
    # sdurmin = transformations[(pool_id,'sun_dur',season, 'obs_min')]
    # sdurmax = transformations[(pool_id,'sun_dur',season, 'obs_max')]
    # sun_dur_p0 = transformations[(pool_id, 'sun_dur', season, 'p0')]
    # sun_dur_a = transformations[(pool_id, 'sun_dur', season, 'a')]
    # sun_dur_b = transformations[(pool_id, 'sun_dur', season, 'b')]
    # sun_dur_loc = transformations[(pool_id, 'sun_dur', season, 'loc')]
    # sun_dur_scale = transformations[(pool_id, 'sun_dur', season, 'scale')]
    
    # --- Initialize arrays for each output_type and variable ---
    for output_type in output_types:
        npt = n_points[output_type]
        for variable in simulation_variables:
            z_scores[(output_type, variable)] = np.zeros((n_days+2, npt))
            values[(output_type, variable)] = np.zeros((n_days+2, npt))
            lag_z_scores[(output_type, variable)] = np.zeros((2, npt))
            lag_values[(output_type, variable)] = np.zeros((2, npt))
        if 'pet' in output_variables:
            values[(output_type, 'pet')] = np.zeros((n_days+2, npt))
    
        for output_type in output_types:
            # Ensure arrays of values are reset to zero
            for variable in simulation_variables:
                z_scores[(output_type, variable)].fill(0.0)
                values[(output_type, variable)].fill(0.0)
            if 'pet' in output_variables:
                values[(output_type, 'pet')].fill(0.0)
        
            # Construct arrays with space for first two lags at beginning (requiring lags from previous months)
            # TODO: Check that lag arrays have first position as lag-1 and second position as lag-2
            for variable in simulation_variables:
                z_scores[(output_type, variable)][0, :] = lag_z_scores[(output_type, variable)][0, :]
                z_scores[(output_type, variable)][1, :] = lag_z_scores[(output_type, variable)][1, :]
                values[(output_type, variable)][0, :] = lag_values[(output_type, variable)][0, :]
                values[(output_type, variable)][1, :] = lag_values[(output_type, variable)][1, :]
        
            # Aggregate input rainfall (current month) to daily timestep
            if timestep != 24:
                # t99a = datetime.datetime.now()
                values[(output_type, 'prcp')][2:,:] = aggregate_rainfall(RAINFALL[realization_counter][output_type], n_points[output_type], int(24 / timestep),)
            else:
                values[(output_type, 'prcp')][2:,:] = RAINFALL[realization_counter][output_type][:]
        
            # Identify transition states
            transitions = np.zeros((n_days,n_points[output_type]), dtype=int)
        
            # Order of assignment such that DDD can overwrite DD
            transitions = np.where(  # DD
                (values[(output_type, 'prcp')][2:, :] < wet_threshold)
                & (values[(output_type, 'prcp')][1:-1, :] < wet_threshold),
                2,
                transitions
            )
            transitions = np.where(  # DDD
                (values[(output_type, 'prcp')][2:, :] < wet_threshold)
                & (values[(output_type, 'prcp')][1:-1, :] < wet_threshold)
                & (values[(output_type, 'prcp')][:-2, :] < wet_threshold),
                1,
                transitions
            )
            transitions = np.where(  # DW
                (values[(output_type, 'prcp')][1:-1, :] < wet_threshold)
                & (values[(output_type, 'prcp')][2:, :] >= wet_threshold),
                3,
                transitions
            )
            transitions = np.where(  # WD
                (values[(output_type, 'prcp')][1:-1, :] >= wet_threshold)
                & (values[(output_type, 'prcp')][2:, :] < wet_threshold),
                4,
                transitions
            )
            transitions = np.where(  # WW
                (values[(output_type, 'prcp')][1:-1, :] >= wet_threshold)
                & (values[(output_type, 'prcp')][2:, :] >= wet_threshold),
                5,transitions)
            variable = 'prcp'   
            if season_length == 'month':
                season = month
                rainfall_mean = interpolated_parameters[('raw_statistics', output_type, variable, season, 'mean')]
                rainfall_stdev = interpolated_parameters[('raw_statistics', output_type, variable, season, 'std')]
                rainfall_sa = (values[(output_type, 'prcp')] - rainfall_mean) / rainfall_stdev
            elif season_length == 'half-month':
                season = (month - 1) * 2 + 1
                rainfall_mean = interpolated_parameters[('raw_statistics', output_type, variable, season, 'mean')]
                rainfall_stdev = interpolated_parameters[('raw_statistics', output_type, variable, season, 'std')]
                rainfall_sa1 = (values[(output_type, 'prcp')] - rainfall_mean) / rainfall_stdev
                season = (month - 1) * 2 + 2
                rainfall_mean = interpolated_parameters[('raw_statistics', output_type, variable, season, 'mean')]
                rainfall_stdev = interpolated_parameters[('raw_statistics', output_type, variable, season, 'std')]
                rainfall_sa2 = (values[(output_type, 'prcp')] - rainfall_mean) / rainfall_stdev
                rainfall_sa = np.zeros(values[(output_type, 'prcp')].shape[0])
                rainfall_sa[:2] = rainfall_sa2[:2]  # TODO: Check that this is done correctly
                rainfall_sa[2:2+15] = rainfall_sa1[2:2+15]  # TODO: Check that this is done correctly
                rainfall_sa[15:] = rainfall_sa2[15:]  # TODO: Check that this is done correctly
    
        z_scores[(output_type, 'prcp')][:] = rainfall_sa[:]
        for k, v in z_scores.items():
            z_scores_nb[k] = v
        
        residuals_dummy = np.zeros(n_points[output_type])
        for variable in input_variables:
            z_scores[(output_type, variable)][2:, :], ri = regressions(n_days, season_length, month, variable, sn_sample, ri, transition_key_nb,
                z_scores_nb, output_type, transitions, parameters_nb, pool_id, predictors_nb,interpolated_parameters_nb, residuals_dummy,)
        
        for variable in input_variables:
            mean_1 = transformed_statistics_dict[(pool_id, variable, season)][0]  # , transition_name
            sd_1 = transformed_statistics_dict[(pool_id, variable, season)][1]  # , transition_name
            values[(output_type, variable)][2:, :] = (z_scores[(output_type, variable)][2:, :] * sd_1 + mean_1)
    
    
        # x = np.arange(0.0, 1.0+0.0001, 0.001)
        # for season in seasons:
        #     y = scipy.stats.beta.ppf(x, sun_dur_a, sun_dur_b, sun_dur_loc, sun_dur_scale)
        #     f = scipy.interpolate.interp1d(x, y, bounds_error=False)
        #     sundur_beta_ppf_funcs[(pool_id, season)] = f
        
        for variable in input_variables:
                values[(output_type, variable)][2:, :] = scipy.special.inv_boxcox(values[(output_type, variable)][2:, :],LAMDF['lamda'][LAMDF['variable']==variable].values)
                offset = offset_season['offset'][offset_season['variable']==variable].values
                values[(output_type, variable)][2:, :] -= offset           
        
        for variable in input_variables:
                mean_2 = IDP_DF['mean'][IDP_DF['variable']==variable].values 
                sd_2 = IDP_DF['sd'][IDP_DF['variable']==variable].values 
                values[(output_type, variable)][2:, :] = (values[(output_type, variable)][2:, :] * sd_2 + mean_2)
            
        
        if variable == 'dtr':
            values[(output_type, variable)] = np.maximum(values[(output_type, variable)], 0.1)
        # elif variable == 'vap_press':
        #     values[(output_type, variable)] = np.maximum(values[(output_type, variable)], 0.01)
        # elif variable == 'wind_speed':
        #     values[(output_type, variable)] = np.maximum(values[(output_type, variable)], 0.01)
        # elif variable == 'sun_dur':
        #     values[(output_type, variable)] = np.maximum(values[(output_type, variable)], 0.0)
        
        lag_z_scores[(output_type, variable)][0, :] = z_scores[(output_type, variable)][n_days+1,:]
        lag_z_scores[(output_type, variable)][1, :] = z_scores[(output_type, variable)][n_days, :]
        lag_values[(output_type, variable)][0, :] = values[(output_type, variable)][n_days+1, :]
        lag_values[(output_type, variable)][1, :] = values[(output_type, variable)][n_days, :]
        if 'pet' in output_variables:
            calculate_pet2(year=year, month = month,output_types = output_types,values = values,discretisation_metadata = discretisation_metadata ,
                          n_points = n_points, latitude = LATITUDE_DEGREES*np.pi/180, wind_height = wind_height)
    
    return values 

def getDates(year,month,SIMLIST):
    '''
    SIMLIST is the output from simulate_daily_weather_point
    '''
    sd = pd.to_datetime(str(year)+'-'+str(month)+'-'+str(1))
    datseq = pd.date_range(sd, periods=len(SIMLIST[0][('point','temp_avg')][2:]))
    return datseq



predictors = {('temp_avg', 'DDD'): ['temp_avg_lag1'],
        ('temp_avg', 'DD'): ['temp_avg_lag1'],
        ('temp_avg', 'DW'): ['temp_avg_lag1', 'prcp'],
        ('temp_avg', 'WD'): ['temp_avg_lag1', 'prcp_lag1'],
        ('temp_avg', 'WW'): ['temp_avg_lag1'],
        ('dtr', 'DDD'): ['dtr_lag1'],
        ('dtr', 'DD'): ['dtr_lag1'],
        ('dtr', 'DW'): ['dtr_lag1', 'prcp'],
        ('dtr', 'WD'): ['dtr_lag1', 'prcp_lag1'],
        ('dtr', 'WW'): ['dtr_lag1'],
        ('vap_press', 'DDD'): ['vap_press_lag1', 'temp_avg', 'dtr'],  # 'prcp',
        ('vap_press', 'DD'): ['vap_press_lag1', 'temp_avg', 'dtr'],  # 'prcp',
        ('vap_press', 'DW'): ['vap_press_lag1', 'prcp', 'temp_avg', 'dtr'],
        ('vap_press', 'WD'): ['vap_press_lag1', 'temp_avg', 'dtr'],  # 'prcp',
        ('vap_press', 'WW'): ['vap_press_lag1', 'prcp', 'temp_avg', 'dtr'],
        ('wind_speed', 'DDD'): ['wind_speed_lag1', 'temp_avg', 'dtr'],  # 'prcp',
        ('wind_speed', 'DD'): ['wind_speed_lag1', 'temp_avg', 'dtr'],  # 'prcp',
        ('wind_speed', 'DW'): ['wind_speed_lag1', 'prcp', 'temp_avg', 'dtr'],
        ('wind_speed', 'WD'): ['wind_speed_lag1', 'temp_avg', 'dtr'],  # 'prcp',
        ('wind_speed', 'WW'): ['wind_speed_lag1', 'prcp', 'temp_avg', 'dtr'],
        ('sun_dur', 'DDD'): ['sun_dur_lag1', 'temp_avg', 'dtr'],  # 'prcp',
        ('sun_dur', 'DD'): ['sun_dur_lag1', 'temp_avg', 'dtr'],  # 'prcp',
        ('sun_dur', 'DW'): ['sun_dur_lag1', 'prcp', 'temp_avg', 'dtr'],
        ('sun_dur', 'WD'): ['sun_dur_lag1', 'temp_avg', 'dtr'],  # 'prcp',
        ('sun_dur', 'WW'): ['sun_dur_lag1', 'prcp', 'temp_avg', 'dtr']}

PARAMS = pd.read_parquet('/home/users/azhar199/DATA/NEWDATA/GRIDDED_WG_QC/HOURLY_DAILY/PARAM5KM/2061_2080_DAILY/UKCP_PARAMS_5KM_2061_2080_DAILY_M04.parquet')

def GetRealizations(RealizationCount,EASTING,NORTHING,RealizationStartYear):
    n_realizations = RealizationCount
    n_years = 20
    base_seed = 42
    rng = np.random.default_rng(seed = base_seed)
    parameters_df = PARAMS[(PARAMS['easting']==EASTING) & (PARAMS['northing']==NORTHING)].reset_index(drop=True)
    def GetMonthLengths2(NUM,START):
        n_years = NUM
        Start = pd.to_datetime(str(START)+str('-01-01'),format='%Y-%m-%d')
        End = pd.to_datetime(str(START+n_years-1)+str('-12-31'),format='%Y-%m-%d')
        DatesVect = pd.date_range(start=Start,end=End,freq="D")
        monthly_df = pd.DataFrame({'Year':DatesVect.year,'Month':DatesVect.month})
        HRCOUNT = monthly_df.groupby(['Year', 'Month']).size().reset_index(name='n_hours')
        return DatesVect,HRCOUNT['n_hours'].to_numpy()
        
        
    month_lengths_array = GetMonthLengths2(n_years,RealizationStartYear)[1]*24
    realizations = []
    
    for i in range(n_realizations):
        rng = np.random.default_rng(seed = base_seed + i)  
        sim_df = main_point_model_monthly(parameters=parameters_df, simulation_length=n_years, month_lengths=month_lengths_array, intensity_distribution='weibull',rng=rng)
        HR = discretise_by_point_monthly(sim_df, simulation_length_years = n_years, STARTYR = RealizationStartYear, timestep_length=24)
        HR_df = pd.DataFrame(HR, columns=['rainfall'])
        realizations.append(HR_df)
    
    
    all_realizations_df = pd.concat(realizations, ignore_index=True,axis=1)
    all_realizations_df.columns = ['Realization'+'_'+str(i+1) for i in range(n_realizations)]
    all_realizations_df['DateTime'] = GetMonthLengths2(n_years,RealizationStartYear)[0]
    all_realizations_df['Year'] = [all_realizations_df['DateTime'][i].year for i in range(all_realizations_df.shape[0])]
    all_realizations_df['Month'] = [all_realizations_df['DateTime'][i].month for i in range(all_realizations_df.shape[0])]
    return all_realizations_df

GRIDS2 = pd.read_csv('/home/users/azhar199/DATA/NEWDATA/GRIDDED_WG_QC/HOURLY_DAILY/HUKMETA2.csv')
easting = GRIDS2['easting'].iloc[3736]
northing = GRIDS2['northing'].iloc[3736]

all_realizations_df = GetRealizations(RealizationCount = 300, EASTING = easting, NORTHING = northing, RealizationStartYear = 2061)

IWS = pd.read_parquet("/home/users/azhar199/DATA/NEWDATA/GRIDDED_WG_QC/HOURLY_DAILY/PARAM5KM/2061_2080_MEMIWS/UKCP_PARAMS_5KM_2061_2080_IWS_FIN_M04.parquet")
raw_statistics = IWS[(IWS['easting']==easting) & (IWS['northing']==northing)].reset_index(drop=True)

r2 = pd.read_parquet("/home/users/azhar199/DATA/NEWDATA/GRIDDED_WG_QC/HOURLY_DAILY/PARAM5KM/WGENPARAMS/R2.parquet")
r2 = r2[(r2['easting']==easting) & (r2['northing']==northing)].reset_index(drop=True)
r2df = r2.drop(['easting','northing'],axis=1)
r2_dict = {(np.float64(row.pool_id),np.int32(row.season),row.variable,row.transition): np.float64(row.r2)  for _, row in r2df.iterrows()}

SER =  pd.read_parquet("/home/users/azhar199/DATA/NEWDATA/GRIDDED_WG_QC/HOURLY_DAILY/PARAM5KM/WGENPARAMS/STANDARD_ERROR.parquet")
SER = SER[(SER['easting']==easting) & (SER['northing']==northing)].reset_index(drop=True)
sedf = SER.drop(['easting','northing'],axis=1)
se_dict = {(np.float64(row.site_id),np.int32(row.month),row.variable,row.transition): np.float64(row.value)  for _, row in sedf.iterrows()}

WGPAR = pd.read_parquet('/home/users/azhar199/DATA/NEWDATA/GRIDDED_WG_QC/HOURLY_DAILY/PARAM5KM/WGENPARAMS/WGEN_REG_PARAMS.parquet')
WGPAR = WGPAR[(WGPAR['easting']==easting) & (WGPAR['northing']==northing)].reset_index(drop=True)
WGPARdf = WGPAR.drop(['easting','northing'],axis=1)
WGPAR_dict = {(int(row.site_id),np.int32(row.month),row.variable,row.transition): row[['beta1', 'beta2', 'beta3']].dropna().to_numpy(dtype=float) for _, row in WGPARdf.iterrows()}

transfix = pd.read_parquet('/home/users/azhar199/DATA/NEWDATA/GRIDDED_WG_QC/HOURLY_DAILY/PARAM5KM/WGENPARAMS/TRANSFORMATION_FIXED.parquet')
transfix = transfix[(transfix['easting']==easting) & (transfix['northing']==northing)].reset_index(drop=True)
transfixdf = transfix.drop(['easting','northing'],axis=1)
transfixdf['value'] = transfixdf['value'].astype(np.float64)
transfix_dict = {(row.pool_id,row.variable,row.season,row.parameter,): np.float64(row.value) for _, row in transfixdf.iterrows()}

transtat = pd.read_parquet('/home/users/azhar199/DATA/NEWDATA/GRIDDED_WG_QC/HOURLY_DAILY/PARAM5KM/WGENPARAMS/TRANSFORMED_STATISTICS.parquet')
transtat = transtat[(transtat['easting']==easting) & (transtat['northing']==northing)].reset_index(drop=True)
transtatdf = transtat.drop(['easting','northing'],axis=1)
transtat_dict = {(row['pool_id'], row['variable'], row['season']): (row['mean'], row['std']) for _, row in transtatdf.iterrows()}

offset = pd.read_parquet('/home/users/azhar199/DATA/NEWDATA/GRIDDED_WG_QC/HOURLY_DAILY/PARAM5KM/WGENPARAMS/OFFSET_DF.parquet')
offset= offset[(offset['easting']==easting) & (offset['northing']==northing)].reset_index(drop=True)
OFFSET_DF = offset.drop(['easting','northing'],axis=1)



def getMonthlySimulations(YearNumber, MonthNumber, LATITUDE_DEGREES, LONGITUDE_DEGREES, point_elevation):
    base_seed = 42
    n_realizations = 300
    simulations = [simulate_daily_weather_point(RAINFALL_REALIZATIONS = all_realizations_df, year = YearNumber, month = MonthNumber, n_realizations = n_realizations, LATITUDE_DEGREES = LATITUDE_DEGREES,
                                                LONGITUDE_DEGREES = LONGITUDE_DEGREES, point_elevation = point_elevation, predictors = predictors, input_variables = ['temp_avg', 'dtr'],
                                                transitions = ['DDD', 'DD', 'DW', 'WD', 'WW'], seasons = list(range(1,13)), raw_statistics = raw_statistics, r2 = r2_dict, standard_errors = se_dict, 
                                                parameters = WGPAR_dict, timestep = 24, output_types = ['point'], n_points = {'point': 1}, transformations = transfix_dict, transformed_statistics_dict = transtat_dict, 
                                                output_variables = ['temp_avg', 'dtr'], wet_threshold = 1, season_length = 'month', wind_height = 2, offset_df = OFFSET_DF, base_seed = base_seed + I, 
                                                realization_counter = I) for I in np.arange(0,n_realizations,1)]

    datesdaily = getDates(year = YearNumber, month = MonthNumber, SIMLIST = simulations)
    temp_avg = pd.DataFrame(np.concatenate([simulations[i][('point', 'temp_avg')][2:] for i in np.arange(0,n_realizations,1)],axis=1),index = datesdaily)
    dtr = pd.DataFrame(np.concatenate([simulations[i][('point', 'dtr')][2:] for i in np.arange(0,n_realizations,1)],axis=1),index = datesdaily)
    prcp = pd.DataFrame(np.concatenate([simulations[i][('point', 'prcp')][2:] for i in np.arange(0,n_realizations,1)],axis=1),index = datesdaily)
    VARLIST = list([temp_avg,dtr,prcp])
    
    for j in np.arange(0,len(VARLIST),1):
        VARLIST[j].columns = ['R_'+str(i) for i in np.arange(1,VARLIST[j].shape[1]+1,1)]
        
    return VARLIST 


def run_year(year):
    months = list(range(1, 13))
    return (year, [getMonthlySimulations(YearNumber=year,
                                         MonthNumber=m,
                                         LATITUDE_DEGREES=convert_lonlat([easting],[northing])[0][0],
                                         LONGITUDE_DEGREES=convert_lonlat([easting],[northing])[1][0],
                                         point_elevation=78) for m in months])

YEARS = np.unique(all_realizations_df['Year'])
YEARSIMS = [None] * len(YEARS)

with ProcessPoolExecutor(max_workers=12) as executor:
    futures = {executor.submit(run_year, y): y for y in YEARS}
    for fut in as_completed(futures):
        year, result = fut.result()
        YEARSIMS[list(YEARS).index(year)] = result



TEMP_AVG = pd.concat([pd.concat([YEARSIMS[i][j][0] for i in range(len(YEARSIMS))], axis=0) for j in range(12)],axis=0)
DTR = pd.concat([pd.concat([YEARSIMS[i][j][1] for j in range(12)]) for i in range(len(YEARSIMS))])
TEMP_AVG_filled = TEMP_AVG.copy()
TEMP_AVG_filled = TEMP_AVG_filled.groupby([TEMP_AVG_filled.index.year, TEMP_AVG_filled.index.month]).transform(lambda x: x.fillna(x.mean()))
DTR_filled = DTR.copy()
DTR_filled = DTR_filled.groupby([DTR_filled.index.year, DTR_filled.index.month]).transform(lambda x: x.fillna(x.mean()))
TEMP_MIN = TEMP_AVG_filled - (0.5*DTR_filled)
TEMP_MAX = TEMP_AVG_filled + (0.5*DTR_filled)
TEMP_AVG_filled.to_csv('/home/users/azhar199/DATA/NEWDATA/GRIDDED_WG_QC/PaperPlots/UsageNotes/TEMPREAL_M04_Exeter.csv',index=False)
DTR_filled.to_csv('/home/users/azhar199/DATA/NEWDATA/GRIDDED_WG_QC/PaperPlots/UsageNotes/DTRREAL_M04_Exeter.csv',index=False)
