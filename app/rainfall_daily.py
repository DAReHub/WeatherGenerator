import pandas as pd
import numpy as np
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import scipy.stats
import scipy.special


def simulate_storms(month_lengths, simulation_length, parameters, rng):
    simulation_end_time = np.cumsum(month_lengths)[-1]

    sim_len_ext = simulation_length + 4
    month_lengths_ext = month_lengths.copy()
    for _ in range(4):
        month_lengths_ext = np.concatenate(
            [month_lengths_ext, month_lengths[-12:]])

    lamda = np.tile(parameters['lamda'].values, sim_len_ext)
    cumulative_expected_storms = np.cumsum(lamda * month_lengths_ext)
    cumulative_month_endtimes = np.cumsum(month_lengths_ext)
    expected_number_of_storms = cumulative_expected_storms[-1]
    number_of_storms = rng.poisson(expected_number_of_storms)

    deformed_arrivals = expected_number_of_storms * np.sort(
        rng.uniform(size=number_of_storms))
    cumulative_expected_storms = np.insert(cumulative_expected_storms, 0, 0.0)
    cumulative_month_endtimes = np.insert(cumulative_month_endtimes, 0, 0.0)
    interpolator = scipy.interpolate.interp1d(
        cumulative_expected_storms, cumulative_month_endtimes
    )
    storm_arrival_times = interpolator(deformed_arrivals)

    storm_arrival_times = storm_arrival_times[
        storm_arrival_times < simulation_end_time]
    number_of_storms = storm_arrival_times.shape[0]
    storms = pd.DataFrame({
        'storm_id': np.arange(number_of_storms),
        'storm_arrival': storm_arrival_times
    })
    storms['month'] = lookup_months(month_lengths, simulation_length,
                                    storms['storm_arrival'].values)
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


def make_storm_arrays_by_raincell(num_cells, storm_ids, storm_arrivals,
                                  storm_months):
    return (
        np.repeat(storm_ids, num_cells),
        np.repeat(storm_arrivals, num_cells),
        np.repeat(storm_months, num_cells)
    )



def main_point_model_monthly(
        parameters,
        simulation_length,
        month_lengths,
        intensity_distribution,
        rng
):
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
    storms, number_of_storms = simulate_storms(month_lengths, simulation_length,
                                               parameters, rng)

    # Step 2 - Simulate raincells
    df = simulate_raincells_point(storms, parameters, rng)

    # Merge parameters into master dataframe
    df = pd.merge(df, parameters, how='left', on='month')

    # Step 3 - Raincell arrival times
    raincell_arrival_times = rng.exponential(
        1.0 / df['beta'])  # relative to storm origin
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


def discretise_point(
        period_start_time,
        timestep_length,
        raincell_arrival_times,
        raincell_end_times,
        raincell_intensities,
        discrete_rainfall
):
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
                discrete_rainfall[
                    timestep, 0] += rc_intensity * timestep_coverage



def discretise_by_point_monthly(df_raincells, simulation_length_years, STARTYR,
                                timestep_length=24.0):
    # Input: timestep_length in hours (default 24 -> produce daily totals)
    Start = pd.to_datetime(str(STARTYR) + str('-01-01 00:00:00'),
                           format='%Y-%m-%d %H:%M:%S')
    End = pd.to_datetime(
        str(STARTYR + simulation_length_years - 1) + str('-12-31 23:00:00'),
        format='%Y-%m-%d %H:%M:%S')
    DatesVect = pd.date_range(start=Start, end=End, freq="D")
    monthly_df = pd.DataFrame(
        {'Year': DatesVect.year, 'Month': DatesVect.month})
    HRCOUNT = monthly_df.groupby(['Year', 'Month']).size().reset_index(
        name='n_days')
    month_days = HRCOUNT['n_days'].to_numpy()

    # number of timesteps per month given timestep_length (hours)
    timesteps_per_month = (month_days * 24 / timestep_length).astype(
        int)  # usually days if timestep_length==24
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
        mask = ((df_raincells['raincell_arrival'] < end_hour) & (
                    df_raincells['raincell_end'] > start_hour))
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


def GetRealizations(
        parameters: pd.DataFrame,
        n_realizations: int,
        EASTING: int,
        NORTHING: int,
        RealizationStartYear: int,
        n_years: int,
        base_seed: int
):
    rng = np.random.default_rng(seed=base_seed)
    parameters_df = parameters[(parameters['easting'] == EASTING) & (
                parameters['northing'] == NORTHING)].reset_index(drop=True)

    def GetMonthLengths2(NUM, START):
        n_years = NUM
        Start = pd.to_datetime(str(START) + str('-01-01'), format='%Y-%m-%d')
        End = pd.to_datetime(str(START + n_years - 1) + str('-12-31'),
                             format='%Y-%m-%d')
        DatesVect = pd.date_range(start=Start, end=End, freq="D")
        monthly_df = pd.DataFrame(
            {'Year': DatesVect.year, 'Month': DatesVect.month})
        HRCOUNT = monthly_df.groupby(['Year', 'Month']).size().reset_index(
            name='n_hours')
        return DatesVect, HRCOUNT['n_hours'].to_numpy()

    month_lengths_array = GetMonthLengths2(n_years, RealizationStartYear)[
                              1] * 24
    realizations = []

    for i in range(n_realizations):
        rng = np.random.default_rng(seed=base_seed + i)
        sim_df = main_point_model_monthly(parameters=parameters_df,
                                          simulation_length=n_years,
                                          month_lengths=month_lengths_array,
                                          intensity_distribution='weibull',
                                          rng=rng)
        HR = discretise_by_point_monthly(sim_df,
                                         simulation_length_years=n_years,
                                         STARTYR=RealizationStartYear,
                                         timestep_length=24)
        HR_df = pd.DataFrame(HR, columns=['rainfall'])
        realizations.append(HR_df)

    all_realizations_df = pd.concat(realizations, ignore_index=True, axis=1)
    all_realizations_df.columns = ['Realization' + '_' + str(i + 1) for i in
                                   range(n_realizations)]
    all_realizations_df['DateTime'] = \
    GetMonthLengths2(n_years, RealizationStartYear)[0]
    all_realizations_df['Year'] = [all_realizations_df['DateTime'][i].year for i
                                   in range(all_realizations_df.shape[0])]
    all_realizations_df['Month'] = [all_realizations_df['DateTime'][i].month for
                                    i in range(all_realizations_df.shape[0])]
    return all_realizations_df
