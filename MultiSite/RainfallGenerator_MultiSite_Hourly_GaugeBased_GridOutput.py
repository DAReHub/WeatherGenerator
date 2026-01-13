import os
import numpy as np
import numpy.ma as ma
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import scipy.stats
from scipy.stats import skew, pearsonr
from scipy import stats
from scipy.optimize import differential_evolution
import itertools
import geocube.api.core
import geocube
import psutil
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import scipy
import scipy.interpolate
import scipy.optimize
import numba
import sys
import datetime
import gstools
import xarray as xr
from netCDF4 import Dataset
import geopandas as gpd


####################################################################################
# The set of functions that follow concern with preparation of spatial time series #
# These functions are taken from analysis.py of rwgen #
def prepare_spatial_timeseries(
        point_metadata,
        timeseries_folder,
        timeseries_format,
        season_definitions,
        completeness_threshold,
        durations,
        outlier_method,
        maximum_relative_difference,
        maximum_alterations,
):
    """
    Prepare timeseries for multiple gauges (spatial model).

    Returns
    -------
    timeseries : dict
        Nested dictionary of the form:
        {point_id: {'1H': df,   # dataframe for 1-hour aggregation
                '24H': df,  # dataframe for 24-hour aggregation
                },
            ...
        }
    """

    timeseries = {}

    # Loop over gauges
    for _, row in point_metadata.iterrows():
        point_id = row['point_id']

        # Pick the file name
        if 'file_name' in point_metadata.columns:
            file_name = row['file_name']
        else:
            file_name = row['name'] + "." + timeseries_format

        input_path = os.path.join(timeseries_folder, file_name)

        # Read input
        if timeseries_format == 'txt':
            df = pd.read_csv(input_path, header=None, names=['value'], dtype={'value': float})
            # You’d need to attach a datetime index if using txt
        elif timeseries_format == 'csv':
            df = pd.read_csv(
                input_path, names=['value'], index_col=0, skiprows=1,
                parse_dates=True, infer_datetime_format=True, dayfirst=True
            )
        else:
            raise ValueError(f"Unsupported timeseries format: {timeseries_format}")

        # Run the point-level preparation
        ts = prepare_point_timeseries(
            df=df,
            season_definitions=season_definitions,
            completeness_threshold=completeness_threshold,
            durations=['1H','24H','72H','1M'],
            outlier_method=outlier_method,
            maximum_relative_difference=maximum_relative_difference,
            maximum_alterations=maximum_alterations,
        )

        timeseries[point_id] = ts

    return timeseries

def prepare_point_timeseries(df, season_definitions, completeness_threshold, durations, outlier_method,maximum_relative_difference, maximum_alterations):
    """
    Prepare point timeseries for analysis.

    Steps are: (1) subset on reference calculation period, (2) define seasons for grouping, (3) applying any trimming
    or clipping to reduce the influence of outliers, and (4) aggregating timeseries to required durations.

    """
    df.index = pd.to_datetime(df.index,format = '%Y-%m-%d %H:%M:%S')
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

        # df1 = df1.to_frame()
        # df1.reset_index(inplace=True)
        # df1.rename(columns={'level_2': 'datetime'}, inplace=True)
        # df1.set_index('datetime', inplace=True)
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


#####################################################################################
# The set of functions that follow concern with computation of reference statistics #
# Certain functions are taken from analysis.py and utils.py of rwgen ##



def nested_dictionary_to_dataframe(wet_threshold):
    dc = {1: {'weight': 3.0, 'duration': '1H', 'name': 'variance'},
    2: {'weight': 3.0, 'duration': '1H', 'name': 'skewness'},
    3: {'weight': 5.0, 'duration': '1H', 'name': 'probability_dry', 'threshold': wet_threshold},
    4: {'weight': 5.0, 'duration': '24H', 'name': 'mean'},
    5: {'weight': 2.0, 'duration': '24H', 'name': 'variance'},
    6: {'weight': 2.0, 'duration': '24H', 'name': 'skewness'},
    7: {'weight': 6.0, 'duration': '24H', 'name': 'probability_dry', 'threshold': wet_threshold},
    8: {'weight': 3.0, 'duration': '24H', 'name': 'autocorrelation', 'lag': 1},
    9: {'weight': 2.0, 'duration': '24H', 'name': 'cross-correlation', 'lag': 0},
    10: {'weight': 3.0, 'duration': '72H', 'name': 'variance'},
    11: {'weight': 0.0, 'duration': '1M', 'name': 'variance'}}

    id_name = 'statistic_id'
    non_id_columns = ['name', 'duration', 'lag', 'threshold', 'weight']
    ids = sorted(list(dc.keys()))
    data = {}
    for non_id_column in non_id_columns:
        data[non_id_column] = []
        for id_ in ids:
            values = dc[id_]
            data[non_id_column].append(values[non_id_column] if non_id_column in values.keys() else 'NA')
    dc1 = {}
    dc1[id_name] = ids
    for non_id_column in non_id_columns:
        dc1[non_id_column] = data[non_id_column]
    df = pd.DataFrame(dc1)
    return df



def GetMonthStats(ListofDFs,NUM,wet_threshold):
    '''
    This function requires outputs from the functions prepare_spatial_timeseries 
    and nested_dictionary_to_dataframe
    '''
    statistic_definitions = nested_dictionary_to_dataframe(wet_threshold)
    statistic_definitions[statistic_definitions['duration']=='1H'].name
    statistic_definitions[statistic_definitions['duration']=='24H'].name
    statistic_definitions[statistic_definitions['duration']=='72H'].name
    statistic_definitions[statistic_definitions['duration']=='1M'].name
    statistic_definitions2 = statistic_definitions.drop(statistic_definitions[(statistic_definitions.name =='cross-correlation')].index)
    
    ListofDFs['1H']['Month'] = [ListofDFs['1H'].index[i].month for i in np.arange(0,ListofDFs['1H'].shape[0],1)]
    ListofDFs['1H']['Year'] = [ListofDFs['1H'].index[i].year for i in np.arange(0,ListofDFs['1H'].shape[0],1)]

    ListofDFs['24H']['Month'] = [ListofDFs['24H'].index[i].month for i in np.arange(0,ListofDFs['24H'].shape[0],1)]
    ListofDFs['24H']['Year'] = [ListofDFs['24H'].index[i].year for i in np.arange(0,ListofDFs['24H'].shape[0],1)]

    ListofDFs['72H']['Month'] = [ListofDFs['72H'].index[i].month for i in np.arange(0,ListofDFs['72H'].shape[0],1)]
    ListofDFs['72H']['Year'] = [ListofDFs['72H'].index[i].year for i in np.arange(0,ListofDFs['72H'].shape[0],1)]

    ListofDFs['1M']['Month'] = [ListofDFs['1M'].index[i].month for i in np.arange(0,ListofDFs['1M'].shape[0],1)]
    ListofDFs['1M']['Year'] = [ListofDFs['1M'].index[i].year for i in np.arange(0,ListofDFs['1M'].shape[0],1)]

    VAR_1H = [np.nanvar(ListofDFs['1H']['value'][ListofDFs['1H']['Month']==i]) for i in range(1,13)]
    SKEW_1H = [skew(ListofDFs['1H']['value'][ListofDFs['1H']['Month'] == i]) for i in range(1, 13)]
    PROB_1H = [len(ListofDFs['1H']['value'][(ListofDFs['1H']['Month'] == i) & (ListofDFs['1H']['value'] < wet_threshold)].values)/len(ListofDFs['1H']['value'][ListofDFs['1H']['Month']==i]) for i in range(1,13)]
    MEAN_24H = [np.nanmean(ListofDFs['24H']['value'][ListofDFs['24H']['Month']==i]) for i in range(1,13)]
    VAR_24H = [np.nanvar(ListofDFs['24H']['value'][ListofDFs['24H']['Month']==i]) for i in range(1,13)]
    SKEW_24H = [skew(ListofDFs['24H']['value'][ListofDFs['24H']['Month'] == i]) for i in range(1, 13)]
    PROB_24H = [len(ListofDFs['24H']['value'][(ListofDFs['24H']['Month'] == i) & (ListofDFs['24H']['value'] < wet_threshold)].values)/len(ListofDFs['24H']['value'][ListofDFs['24H']['Month']==i]) for i in range(1,13)]

    def getacf(MONTHNUM):
        df=pd.DataFrame({'x': ListofDFs['24H']['value'][ListofDFs['24H']['Month']==MONTHNUM], 'x_lag': ListofDFs['24H']['value'][ListofDFs['24H']['Month']==MONTHNUM].shift(1)})
        df.dropna(inplace=True)
        acf,pval = scipy.stats.pearsonr(df['x'], df['x_lag'])
        return acf
 
    ACF_24H=[getacf(i) for i in range(1,13)]
    VAR_72H = [np.nanvar(ListofDFs['72H']['value'][ListofDFs['72H']['Month']==i]) for i in range(1,13)]
    VAR_1M = [np.nanvar(ListofDFs['1M']['value'][ListofDFs['1M']['Month']==i]) for i in range(1,13)]

    # standardization of the statistics
    STAN_VAR_1H = np.mean([np.nanvar(ListofDFs['1H']['value'][ListofDFs['1H']['Year']==i]) for i in np.unique(ListofDFs['1H']['Year'])])
    STAN_SKEW_1H = np.mean([skew(ListofDFs['1H']['value'][ListofDFs['1H']['Year']==i], nan_policy='omit') for i in np.unique(ListofDFs['1H']['Year'])])
    STAN_MEAN_24H = np.mean([np.nanmean(ListofDFs['24H']['value'][ListofDFs['24H']['Year']==i]) for i in np.arange(min(ListofDFs['24H']['Year']),max(ListofDFs['24H']['Year'])+1,1)])
    STAN_VAR_24H = np.mean([np.nanvar(ListofDFs['24H']['value'][ListofDFs['24H']['Year']==i]) for i in np.unique(ListofDFs['24H']['Year'])])
    STAN_SKEW_24H = np.mean([skew(ListofDFs['24H']['value'][ListofDFs['24H']['Year']==i], nan_policy='omit') for i in np.unique(ListofDFs['24H']['Year'])])
    STAN_VAR_72H = np.mean([np.nanvar(ListofDFs['72H']['value'][ListofDFs['72H']['Year']==i]) for i in np.unique(ListofDFs['72H']['Year'])])
    STAN_VAR_1M = np.mean([np.nanvar(ListofDFs['1M']['value'][ListofDFs['1M']['Year']==i]) for i in np.unique(ListofDFs['1M']['Year'])])

    STAT =  pd.DataFrame({'point_id':NUM,'statistic_id':np.repeat(statistic_definitions2['statistic_id'],12),
       'name':np.repeat(statistic_definitions2['name'],12),
       'duration':np.repeat(statistic_definitions2['duration'],12),
       'month':np.tile(range(1,13),statistic_definitions2.shape[0]),
       'value':np.concatenate((VAR_1H,SKEW_1H,PROB_1H,MEAN_24H,VAR_24H,SKEW_24H,PROB_24H,ACF_24H,VAR_72H,VAR_1M)),
       'weight':np.repeat(statistic_definitions2['weight'],12),
       'gs':np.concatenate((np.repeat(STAN_VAR_1H,12),np.repeat(STAN_SKEW_1H,12),np.repeat(1,12),np.repeat(STAN_MEAN_24H,12),np.repeat(STAN_VAR_24H,12),
            np.repeat(STAN_SKEW_24H,12),np.repeat(1,24),np.repeat(STAN_VAR_72H,12),np.repeat(STAN_VAR_1M,12))),
       'phi':np.tile(np.array(MEAN_24H)/3,statistic_definitions2.shape[0]),'point_id2':pd.NA,'distance':pd.NA,'phi2':pd.NA})

    STAT.loc[STAT['name'].str.contains('autocorrelation'), 'lag'] = 1
    STAT.loc[STAT['name'].str.contains('probability_dry'), 'threshold'] = wet_threshold
    return STAT


def GetPooledMonthStats(ListofDFs,wet_threshold):
    statistic_definitions = nested_dictionary_to_dataframe(wet_threshold)
    statistic_definitions[statistic_definitions['duration']=='1H'].name
    statistic_definitions[statistic_definitions['duration']=='24H'].name
    statistic_definitions[statistic_definitions['duration']=='72H'].name
    statistic_definitions[statistic_definitions['duration']=='1M'].name
    statistic_definitions2 = statistic_definitions.drop(statistic_definitions[(statistic_definitions.name =='cross-correlation')].index)

    for i in np.arange(1,len(ListofDFs )+1,1):
        ListofDFs [i]['1H']['Month'] = [ListofDFs [i]['1H'].index[j].month for j in np.arange(0,ListofDFs [i]['1H'].shape[0],1)]
        ListofDFs [i]['1H']['Year'] = [ListofDFs [i]['1H'].index[j].year for j in np.arange(0,ListofDFs [i]['1H'].shape[0],1)]
        ListofDFs [i]['24H']['Month'] = [ListofDFs [i]['24H'].index[j].month for j in np.arange(0,ListofDFs [i]['24H'].shape[0],1)]
        ListofDFs [i]['24H']['Year'] = [ListofDFs [i]['24H'].index[j].year for j in np.arange(0,ListofDFs [i]['24H'].shape[0],1)]
        ListofDFs [i]['72H']['Month'] = [ListofDFs [i]['72H'].index[j].month for j in np.arange(0,ListofDFs [i]['72H'].shape[0],1)]
        ListofDFs [i]['72H']['Year'] = [ListofDFs [i]['72H'].index[j].year for j in np.arange(0,ListofDFs [i]['72H'].shape[0],1)]
        ListofDFs [i]['1M']['Month'] = [ListofDFs [i]['1M'].index[j].month for j in np.arange(0,ListofDFs [i]['1M'].shape[0],1)]
        ListofDFs [i]['1M']['Year'] = [ListofDFs [i]['1M'].index[j].year for j in np.arange(0,ListofDFs [i]['1M'].shape[0],1)]

    ScaledList_Daily = []
    for L in np.arange(1,len(ListofDFs )+1,1):
        means = [np.nanmean(ListofDFs [L]['24H']['value'][ListofDFs [L]['24H']['Month']==i]) for i in range(1,13)] 
        Scaled = [ListofDFs [L]['24H']['value'][ListofDFs [L]['24H']['Month']==i]*(3/np.array(means)[i-1]) for i in np.arange(1,13,1)]
        ScaledList_Daily.append(Scaled)
    
    MONLIST = [pd.concat([ScaledList_Daily[u][j] for u in np.arange(0,len(ScaledList_Daily),1)],axis=0) for j in np.arange(0,12,1)]
    MEAN_24H = [np.nanmean(MONLIST[i]) for i in np.arange(0,12,1)]
    VAR_24H = [np.nanvar(MONLIST[i]) for i in np.arange(0,12,1)]
    SKEW_24H = [skew(MONLIST[i]) for i in np.arange(0,12,1)]
    PROB_24H = [len(MONLIST[i].values[MONLIST[i].values<0.2])/len(MONLIST[i].values) for i in np.arange(0,12,1)]

    def getacf(MONTHNUM): 
        df=pd.DataFrame({'x': MONLIST[MONTHNUM], 'x_lag': MONLIST[MONTHNUM].shift(1)})
        df.dropna(inplace=True)
        acf,pval = scipy.stats.pearsonr(df['x'], df['x_lag'])
        return acf
    
    ACF_24H=[getacf(i) for i in range(0,12)]
    ScaledList_Hourly,ScaledList_72H,ScaledList_1M = [[] for i in range(3)]
    for L in np.arange(1,len(ListofDFs )+1,1): 
        means2 = [np.nanmean(ListofDFs [L]['24H']['value'][ListofDFs [L]['24H']['Month']==i]) for i in range(1,13)] 
        Scaled2 = [ListofDFs [L]['1H']['value'][ListofDFs [L]['1H']['Month']==i]*(3/np.array(means2)[i-1]) for i in np.arange(1,13,1)]
        Scaled3 = [ListofDFs [L]['72H']['value'][ListofDFs [L]['72H']['Month']==i]*(3/np.array(means2)[i-1]) for i in np.arange(1,13,1)]
        Scaled4 = [ListofDFs [L]['1M']['value'][ListofDFs [L]['1M']['Month']==i]*(3/np.array(means2)[i-1]) for i in np.arange(1,13,1)]
        ScaledList_Hourly.append(Scaled2)
        ScaledList_72H.append(Scaled3)
        ScaledList_1M.append(Scaled4)
    
    MONLIST_HR = [pd.concat([ScaledList_Hourly[u][j] for u in np.arange(0,len(ScaledList_Hourly),1)],axis=0) for j in np.arange(0,12,1)]
    MONLIST_72H = [pd.concat([ScaledList_72H[u][j] for u in np.arange(0,len(ScaledList_72H),1)],axis=0) for j in np.arange(0,12,1)]
    MONLIST_1M = [pd.concat([ScaledList_1M[u][j] for u in np.arange(0,len(ScaledList_1M),1)],axis=0) for j in np.arange(0,12,1)]
    VAR_1H = [np.nanvar(MONLIST_HR[i]) for i in np.arange(0,12,1)]
    SKEW_1H = [skew(MONLIST_HR[i]) for i in range(0, 12)]
    PROB_1H = [len(MONLIST_HR[i].values[MONLIST_HR[i].values<wet_threshold])/len(MONLIST_HR[i].values) for i in np.arange(0,12,1)]
    VAR_72H = [np.nanvar(MONLIST_72H[i]) for i in np.arange(0,12,1)]
    VAR_1M = [np.nanvar(MONLIST_1M[i]) for i in np.arange(0,12,1)]
    DailyScaledDF = pd.DataFrame(pd.concat(MONLIST,axis=0))
    DailyScaledDF['Year'] = [DailyScaledDF.index[0].year for i in np.arange(0,len(DailyScaledDF),1)]

    H1ScaledDF = pd.DataFrame(pd.concat(MONLIST_HR,axis=0))
    H1ScaledDF['Year'] = [H1ScaledDF.index[0].year for i in np.arange(0,len(H1ScaledDF),1)]

    H72ScaledDF = pd.DataFrame(pd.concat(MONLIST_72H,axis=0))
    H72ScaledDF['Year'] = [H72ScaledDF.index[0].year for i in np.arange(0,len(H72ScaledDF),1)]

    H1MScaledDF = pd.DataFrame(pd.concat(MONLIST_1M,axis=0))
    H1MScaledDF['Year'] = [H1MScaledDF.index[0].year for i in np.arange(0,len(H1MScaledDF),1)]

    STAN_VAR_1H =  np.mean([np.nanvar(H1ScaledDF['value'][H1ScaledDF['Year']==i]) for i in np.unique(H1ScaledDF['Year'])])
    STAN_SKEW_1H = np.mean([skew(H1ScaledDF['value'][H1ScaledDF['Year']==i], nan_policy='omit') for i in np.unique(H1ScaledDF['Year'])])
    STAN_MEAN_24H = np.mean([np.nanmean(DailyScaledDF['value'][DailyScaledDF['Year']==i]) for i in np.unique(DailyScaledDF['Year'])])
    STAN_VAR_24H = np.mean([np.nanvar(DailyScaledDF['value'][DailyScaledDF['Year']==i]) for i in np.unique(DailyScaledDF['Year'])])
    STAN_SKEW_24H = np.mean([skew(DailyScaledDF['value'][DailyScaledDF['Year']==i]) for i in np.unique(DailyScaledDF['Year'])])
    STAN_VAR_72H = np.mean([np.nanvar(H72ScaledDF['value'][H72ScaledDF['Year']==i]) for i in np.unique(H72ScaledDF['Year'])])
    STAN_VAR_1M = np.mean([np.nanvar(H1MScaledDF['value'][H1MScaledDF['Year']==i]) for i in np.unique(H1MScaledDF['Year'])])

    STAT =  pd.DataFrame({'point_id':-1,'statistic_id':np.repeat(statistic_definitions2['statistic_id'],12),
    'name':np.repeat(statistic_definitions2['name'],12),
    'duration':np.repeat(statistic_definitions2['duration'],12),
    'month':np.tile(range(1,13),statistic_definitions2.shape[0]),
    'value':np.concatenate((VAR_1H,SKEW_1H,PROB_1H,MEAN_24H,VAR_24H,SKEW_24H,PROB_24H,ACF_24H,VAR_72H,VAR_1M)),
    'weight':np.repeat(statistic_definitions2['weight'],12),
    'gs':np.concatenate((np.repeat(STAN_VAR_1H,12),np.repeat(STAN_SKEW_1H,12),np.repeat(1,12),np.repeat(STAN_MEAN_24H,12),np.repeat(STAN_VAR_24H,12),
    np.repeat(STAN_SKEW_24H,12),np.repeat(1,24),np.repeat(STAN_VAR_72H,12),np.repeat(STAN_VAR_1M,12))),
    'phi':np.repeat(1,statistic_definitions2.shape[0]*12),'point_id2':pd.NA,'distance':pd.NA,'phi2':pd.NA})

    STAT.loc[STAT['name'].str.contains('autocorrelation'), 'lag'] = 1
    STAT.loc[STAT['name'].str.contains('probability_dry'), 'threshold'] = wet_threshold
    return STAT
    

def GetInterGaugeDistances(METADATA_FILEPATH):
    metadata = pd.read_csv(METADATA_FILEPATH)
    pairs = list(itertools.combinations(list(metadata['point_id']), 2))
    id1s,id2s,distances = [[] for i in range(3)]
    for id1, id2 in pairs:
        id1_x = metadata.loc[metadata['point_id'] == id1, 'easting'].values[0]
        id1_y = metadata.loc[metadata['point_id'] == id1, 'northing'].values[0]
        id2_x = metadata.loc[metadata['point_id'] == id2, 'easting'].values[0]
        id2_y = metadata.loc[metadata['point_id'] == id2, 'northing'].values[0]
        distance = ((id1_x - id2_x) ** 2 + (id1_y - id2_y) ** 2) ** 0.5
        id1s.append(id1)
        id2s.append(id2)
        distances.append(distance / 1000.0)  # m to km
    pair_metadata = pd.DataFrame({'point_id': id1s, 'point_id2': id2s, 'distance': distances})
    return pair_metadata



def GetCrossCorrel(DFLIST,IntergDF,MONTH):
    '''
    DFLIST is ALLDF
    IntergDF = GetInterGaugeDistances(METADATA_FILEPATH)
    '''
    cor=[] 
    for i in np.arange(0,IntergDF.shape[0],1): 
        id1 = IntergDF['point_id'][i] 
        id2 = IntergDF['point_id2'][i]
        DIST = IntergDF['distance'][i]
        lag=0
        df1 = DFLIST[id1]['24H']
        df2 = DFLIST[id2]['24H']
        # 
        x = df1.loc[df1.index.month == MONTH]
        y = df2.loc[df2.index.month == MONTH]
        df3 = pd.merge(x, y, left_index=True, right_index=True)
        r, p = scipy.stats.pearsonr(df3['value_x'].iloc[lag:],df3['value_y'].shift(lag).iloc[lag:])
        MEAN_24H = np.nanmean(df1['value'][df1.index.month == MONTH]) / 3
        MEAN_24H_P2 = np.nanmean(df2['value'][df2.index.month == MONTH]) / 3
        cordf = pd.DataFrame([{'point_id':id1,'statistic_id':9, 'name':'cross-correlation_lag0', 'duration':'24H', 'month':MONTH, 'value':r,'weight':2, 'gs':1, 'phi':MEAN_24H, 'point_id2':id2, 'distance':DIST, 'phi2':MEAN_24H_P2}])
        cor.append(cordf)
    CORDF = pd.concat(cor,axis=0) 
    return CORDF


def exponential_model(distance, variance, length_scale, nugget=None):
    if nugget is None:
        _nugget = 1.0 - variance
    else:
        _nugget = nugget
    x = variance * np.exp(-distance / length_scale) + _nugget
    return x


def get_fitted_correlations(df, unique_months):  # df = cc2
     bounds = ([0.01, 0.0], [1.0, 100000000.0])
     tmp = []
     for month in unique_months:
         parameters, _ = scipy.optimize.curve_fit(
             exponential_model,
             df.loc[df['month'] == month, 'distance'].values,
             df.loc[df['month'] == month, 'value'].values,
             bounds=bounds
         )
         variance, length_scale = parameters
         _corrs = exponential_model(df.loc[df['month'] == month, 'distance'].values, variance, length_scale,)
         df1 = pd.DataFrame({'distance': df.loc[df['month'] == month, 'distance'].values,'value': _corrs})
         df1['month'] = month
         tmp.append(df1)
     df1 = pd.concat(tmp)
     df1 = df.drop(columns='value').merge(df1)
     return df1
 

def GetPoolCrossCorrel(CROSSDF):
    '''
    This function requires output of GetCrossCorrel(MONTH) as its input and two other functions should be formulated
    which are exponential_model and get_fitted_correlations
    ''' 
    nbins = min(CROSSDF.loc[(CROSSDF['month'] == 1)].shape[0], 20)
    distance_bins = np.linspace(CROSSDF['distance'].min() - 0.001,CROSSDF['distance'].max() + 0.001,nbins) # min and max ig distances
    bin_midpoints = (np.concatenate([np.array([0.0]), distance_bins[:-1]]) + distance_bins) / 2.0 # these are the midpoint distances
    CROSSDF['distance_bin']=np.digitize(CROSSDF['distance'], distance_bins)
    cc2 = CROSSDF.groupby(['month', 'distance_bin'])['value'].mean()
    cc2 = cc2.to_frame('value')
    cc2.reset_index(inplace=True)
    tmp = pd.DataFrame({'distance_bin': np.arange(bin_midpoints.shape[0], dtype=int), 'distance': bin_midpoints})
    cc2 = cc2.merge(tmp)
    cc2.sort_values(['distance_bin', 'month'], inplace=True)
    cc3 = get_fitted_correlations(cc2,list(range(1, 13)))
    cc3.sort_values(['distance_bin', 'month'], inplace=True)
    poolcross=pd.DataFrame({'point_id':-1, 'statistic_id':9, 'name':'cross-correlation_lag0', 'duration':'24H', 'month':cc3['month'], 'value':cc3['value'],'weight':2, 'gs':1, 'phi':1, 'point_id2':-1, 'distance':cc3['distance'], 'phi2':1})
    return poolcross



###############################################################################
# The following functions pertain to fitting the NSRP_spatial parameters #######

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

        # -----------------------
        #  Fixed parameters check
        # -----------------------
        if not isinstance(fixed_parameters, dict):
            if (
                (("rho" in fixed_parameters.columns) and ("gamma" not in fixed_parameters.columns))
                or (("gamma" in fixed_parameters.columns) and ("rho" not in fixed_parameters.columns))
            ):
                raise ValueError("Both rho and gamma must be fixed (or neither fixed).")

        # -----------------------
        #  Prepare statistics
        # -----------------------
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

        # -----------------------
        #  Parameter setup
        # -----------------------
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

        # -----------------------
        #  Initial guess
        # -----------------------
        x0 = initial_parameters.get(month) if initial_parameters is not None else None

        # -----------------------
        #  Optimisation (stage 1)
        # -----------------------
        result = scipy.optimize.differential_evolution(
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

        # Burton et al. (2008) equation 10
        if threshold == 0.1:
            corr_pdry = 0.114703 + 0.884491 * uncorr_pdry

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



##################################################################################################################
# The following functions pertain to NSRP simulation using the parameters obtained from the fitting module #######


def simulate_storms(month_lengths, simulation_length, parameters, rng):
    """
    Simulate storms as a temporal Poisson process.

    """
    # Ensure that Poisson process is sampled beyond end of simulation to avoid any truncation errors
    simulation_end_time = np.cumsum(month_lengths)[-1]
    simulation_length = simulation_length
    month_lengths = month_lengths

    while True:

        # Set up simulation_length and month_lengths with buffer applied
        simulation_length += 4
        for _ in range(4):
            month_lengths = np.concatenate([month_lengths, month_lengths[-12:]])

        # Repeat each set of monthly lamda values for each year in simulation
        lamda = np.tile(parameters['lamda'].values, simulation_length)

        # Get a sample value for number of storms given simulation length
        cumulative_expected_storms = np.cumsum(lamda * month_lengths)
        cumulative_month_endtimes = np.cumsum(month_lengths)
        expected_number_of_storms = cumulative_expected_storms[-1]
        number_of_storms = rng.poisson(expected_number_of_storms)  # sampled

        # Sample storm arrival times on deformed timeline
        deformed_storm_arrival_times = (expected_number_of_storms * np.sort(rng.uniform(size=number_of_storms)))

        # Transform storm origin times from deformed to linear timeline
        cumulative_expected_storms = np.insert(cumulative_expected_storms, 0, 0.0)
        cumulative_month_endtimes = np.insert(cumulative_month_endtimes, 0, 0.0)
        interpolator = scipy.interpolate.interp1d(cumulative_expected_storms, cumulative_month_endtimes)
        storm_arrival_times = interpolator(deformed_storm_arrival_times)

        # Terminate sampling process once a storm has been simulated beyond the simulation end time (and then restrict
        # to just those storms arriving before the end time)
        if storm_arrival_times[-1] > simulation_end_time:
            storm_arrival_times = storm_arrival_times[storm_arrival_times < simulation_end_time]
            number_of_storms = storm_arrival_times.shape[0]
            storms = pd.DataFrame({'storm_id': np.arange(number_of_storms),'storm_arrival': storm_arrival_times})
            storms['month'] = lookup_months(month_lengths, simulation_length, storms['storm_arrival'].values)
            break

    return storms, number_of_storms

def lookup_months(month_lengths, period_length, times):
    end_times = np.cumsum(month_lengths)
    repeated_months = np.tile(np.arange(1, 13, dtype=int), period_length)
    idx = np.digitize(times, end_times)
    return repeated_months[idx]


def simulate_raincells_spatial(storms, parameters, xmin, xmax, ymin, ymax, xrange, yrange, rng, buffer, buffer_factor):
    """
    Simulate raincells for spatial model.  # TODO: Expand explanation

    Notes:
        Requires parameters dataframe to be ordered by month (1-12).

    """
    # Loop is by unique month (1-12)
    i = 0
    for _, row in parameters.iterrows():  # could be replaced by a loop through range(1, 12+1)
        storms_in_month = storms.loc[storms['month'] == row['month']]
        month_number_of_storms = storms_in_month.shape[0]
        month_number_of_raincells_by_storm, \
            month_raincell_x_coords, \
            month_raincell_y_coords, \
            month_raincell_radii = (
                simulate_raincells_for_month(
                    row['rho'], row['gamma'], month_number_of_storms, xmin, xmax, ymin, ymax, xrange, yrange, rng,
                    buffer, buffer_factor
                )
            )

        # Associate parent storm properties with each raincell
        month_storm_ids_by_raincell, month_storm_arrivals_by_raincell, _ = make_storm_arrays_by_raincell(
            month_number_of_raincells_by_storm, storms_in_month['storm_id'].values,
            storms_in_month['storm_arrival'].values, storms_in_month['month'].values
        )

        # Concatenate the arrays (appending to first month processed)
        if i == 0:
            number_of_raincells_by_storm = month_number_of_raincells_by_storm
            raincell_x_coords = month_raincell_x_coords
            raincell_y_coords = month_raincell_y_coords
            raincell_radii = month_raincell_radii
            storm_ids_by_raincell = month_storm_ids_by_raincell
            storm_arrivals_by_raincell = month_storm_arrivals_by_raincell
            months_by_raincell = np.zeros(month_storm_ids_by_raincell.shape[0]) + int(row['month'])
        else:
            number_of_raincells_by_storm = np.concatenate([
                number_of_raincells_by_storm, month_number_of_raincells_by_storm
            ])
            raincell_x_coords = np.concatenate([raincell_x_coords, month_raincell_x_coords])
            raincell_y_coords = np.concatenate([raincell_y_coords, month_raincell_y_coords])
            raincell_radii = np.concatenate([raincell_radii, month_raincell_radii])
            storm_ids_by_raincell = np.concatenate([storm_ids_by_raincell, month_storm_ids_by_raincell])
            storm_arrivals_by_raincell = np.concatenate([
                storm_arrivals_by_raincell, month_storm_arrivals_by_raincell
            ])
            months_by_raincell = np.concatenate([
                months_by_raincell, np.zeros(month_storm_ids_by_raincell.shape[0]) + int(row['month'])
            ])
        i += 1

    # Put into dataframe and then sort
    df = pd.DataFrame({
        'storm_id': storm_ids_by_raincell,
        'storm_arrival': storm_arrivals_by_raincell,
        'month': months_by_raincell,
        'raincell_x': raincell_x_coords,
        'raincell_y': raincell_y_coords,
        'raincell_radii': raincell_radii,
    })
    df.sort_values('storm_arrival', inplace=True)

    return df



def simulate_raincells_for_month(rho, gamma, number_of_storms, xmin, xmax, ymin, ymax, xrange, yrange, rng, buffer, buffer_factor):
    """
    Simulate raincells in inner and outer regions of domain for a calendar month (e.g. all Januarys).

    """
    # Inner region - "standard" spatial Poisson process
    # inner_number_of_raincells_by_storm = rng.poisson(rho * area, number_of_storms)
    # inner_number_of_raincells = np.sum(inner_number_of_raincells_by_storm)
    # inner_x_coords = rng.uniform(xmin, xmax, inner_number_of_raincells)
    # inner_y_coords = rng.uniform(ymin, ymax, inner_number_of_raincells)
    # inner_radii = rng.exponential((1.0 / gamma), inner_number_of_raincells)
    inner_number_of_raincells_by_storm,\
        inner_number_of_raincells,\
        inner_x_coords,\
        inner_y_coords,\
        inner_radii = spatial_poisson_process(
            rho, gamma, number_of_storms, xmin, xmax, ymin, ymax, rng, buffer, buffer_factor
        )

    # Simulate outer region if using Burton et al. (2010) method
    # TODO: Fix this method - it does not work properly yet!
    if not buffer:

        # Construct CDF lookup function for distances of relevant raincells occurring in outer
        # region - Burton et al. (2010) equation A8
        distance_from_quantile_func = construct_outer_raincells_inverse_cdf(gamma, xrange, yrange)

        # Density of relevant raincells in outer region - Burton et al. (2010) equation A9
        rho_y = 2 * rho / gamma ** 2 * (gamma * (xrange + yrange) + 4)

        # Number of relevant raincells in outer region
        outer_number_of_raincells_by_storm = rng.poisson(rho_y, number_of_storms)  # check rho=mean
        outer_number_of_raincells = np.sum(outer_number_of_raincells_by_storm)

        # Sample from CDF of distances of relevant raincells occurring in outer region
        outer_raincell_distance_quantiles = rng.uniform(0.0, 1.0, outer_number_of_raincells)
        outer_raincell_distances = distance_from_quantile_func(outer_raincell_distance_quantiles)

        # Sample eastings and northings from uniform distribution given distance from domain
        # boundaries
        outer_x_coords, outer_y_coords = sample_outer_locations(
            outer_raincell_distances, xrange, yrange, xmin, xmax, ymin, ymax, rng
        )

        # Sample raincell radii - for outer region raincells the radii need to exceed the distance
        # of the cell centre from the domain boundary (i.e. conditional)
        min_quantiles = scipy.stats.expon.cdf(outer_raincell_distances, scale=(1.0 / gamma))
        quantiles = rng.uniform(min_quantiles, np.ones(min_quantiles.shape[0]))
        outer_radii = scipy.stats.expon.ppf(quantiles, scale=(1.0 / gamma))

        # Combine inner and outer region raincells
        # number_of_raincells_by_storm = inner_number_of_raincells_by_storm + outer_number_of_raincells_by_storm
        # raincell_x_coords = np.concatenate([inner_x_coords, outer_x_coords])
        # raincell_y_coords = np.concatenate([inner_y_coords, outer_y_coords])
        # raincell_radii = np.concatenate([inner_radii, outer_radii])
        # TODO: Remove four commented out lines above (incorrect - newer lines below are correct)
        number_of_raincells_by_storm,\
            raincell_x_coords,\
            raincell_y_coords,\
            raincell_radii = combine_inner_outer_raincells(
                inner_number_of_raincells_by_storm, outer_number_of_raincells_by_storm, inner_x_coords, outer_x_coords,
                inner_y_coords, outer_y_coords, inner_radii, outer_radii
            )

    else:
        number_of_raincells_by_storm = inner_number_of_raincells_by_storm
        raincell_x_coords = inner_x_coords
        raincell_y_coords = inner_y_coords
        raincell_radii = inner_radii

    return number_of_raincells_by_storm, raincell_x_coords, raincell_y_coords, raincell_radii


def spatial_poisson_process(
        rho, gamma, number_of_storms, xmin, xmax, ymin, ymax, rng, buffer=True, buffer_factor=15
):
    # Apply buffer to domain
    if buffer:
        radius_variance = scipy.stats.expon.stats(moments='v', scale=(1.0 / gamma))
        buffer_distance = buffer_factor * radius_variance ** 0.5
    else:
        buffer_distance = 0.0
    xmin_b = xmin - buffer_distance
    xmax_b = xmax + buffer_distance
    ymin_b = ymin - buffer_distance
    ymax_b = ymax + buffer_distance
    area_b = (xmax_b - xmin_b) * (ymax_b - ymin_b)

    # Simulate raincells (number, location and radii)
    n_raincells_by_storm = rng.poisson(rho * area_b, number_of_storms)
    n_raincells = np.sum(n_raincells_by_storm)
    x_coords = rng.uniform(xmin_b, xmax_b, n_raincells)
    y_coords = rng.uniform(ymin_b, ymax_b, n_raincells)
    radii = rng.exponential((1.0 / gamma), n_raincells)

    # Remove irrelevant raincells (and update n_raincells_by_storm)
    storm_ids_by_raincell = np.repeat(np.arange(number_of_storms, dtype=int), n_raincells_by_storm)
    relevant_flag = find_relevant_raincells(x_coords, y_coords, radii, xmin, xmax, ymin, ymax)
    df = pd.DataFrame({'storm_id': storm_ids_by_raincell, 'relevant': relevant_flag})
    df = df.groupby(['storm_id'])['relevant'].sum()
    n_raincells_by_storm = df.values
    n_raincells = np.sum(n_raincells_by_storm)
    x_coords = x_coords[relevant_flag]
    y_coords = y_coords[relevant_flag]
    radii = radii[relevant_flag]

    return n_raincells_by_storm, n_raincells, x_coords, y_coords, radii



def sample_outer_locations(d, xrange, yrange, xmin, xmax, ymin, ymax, rng):
    """
    Sample centre locations of raincells in outer region given their distances (d) from the domain boundary.

    """
    # d = distance to raincell centre = x in Burton et al. (2010)
    # vectorised so perimeter array contains a perimeter for each raincell's distance d

    # Perimeter is the sum of the domain perimeter and four quarter-circle arc lengths
    perimeter = 2 * xrange + 2 * yrange
    perimeter += 2 * np.pi * d

    # Sample along the perimeter
    uniform_sample = rng.uniform(0.0, 1.0, perimeter.shape[0])
    position_1d = uniform_sample * perimeter

    # Identify which of the eight line segments that the sampled lengths correspond to using the lower left as a
    # reference point (xmin-d, ymin). Also identify the length relative to the segment origin (first point reached
    # moving clockwise from lower left)
    corner_length = (2.0 * np.pi * d) / 4.0  # quarter-circle arc length
    segment_id = np.zeros(perimeter.shape[0], dtype=int)
    segment_position = np.zeros(perimeter.shape[0])  # i.e. length relative to segment origin
    for i in range(1, 8+1):
        if i == 1:
            min_length = np.zeros(perimeter.shape[0])
            max_length = xrange
        elif i == 2:
            min_length = np.zeros(perimeter.shape[0]) + xrange
            max_length = xrange + corner_length
        elif i == 3:
            min_length = xrange + corner_length
            max_length = xrange + corner_length + yrange
        elif i == 4:
            min_length = xrange + corner_length + yrange
            max_length = xrange + 2 * corner_length + yrange
        elif i == 5:
            min_length = xrange + 2 * corner_length + yrange
            max_length = 2 * xrange + 2 * corner_length + yrange
        elif i == 6:
            min_length = 2 * xrange + 2 * corner_length + yrange
            max_length = 2 * xrange + 3 * corner_length + yrange
        elif i == 7:
            min_length = 2 * xrange + 3 * corner_length + yrange
            max_length = 2 * xrange + 3 * corner_length + 2 * yrange
        elif i == 8:
            min_length = 2 * xrange + 3 * corner_length + 2 * yrange
            max_length = perimeter  # = 2 * xrange + 4 * corner_length + 2 * yrange

        segment_id[(position_1d >= min_length) & (position_1d < max_length)] = i

        segment_position[segment_id == i] = (position_1d[segment_id == i] - min_length[segment_id == i])

    # Identify eastings and northings for straight-line segments first (1, 3, 5, 7)
    x = np.zeros(perimeter.shape[0])
    y = np.zeros(perimeter.shape[0])
    x[segment_id == 1] = xmin - d[segment_id == 1]
    y[segment_id == 1] = ymin + segment_position[segment_id == 1]
    x[segment_id == 3] = xmin + segment_position[segment_id == 3]
    y[segment_id == 3] = ymax + d[segment_id == 3]
    x[segment_id == 5] = xmax + d[segment_id == 5]
    y[segment_id == 5] = ymax - segment_position[segment_id == 5]
    x[segment_id == 7] = xmax - segment_position[segment_id == 7]
    y[segment_id == 7] = ymin - d[segment_id == 7]

    # Identify eastings and northings for corner segments (2, 4, 6, 8)
    theta = np.zeros(perimeter.shape[0])  # angle of sector corresponding with arc length

    theta[segment_id == 2] = segment_position[segment_id == 2] / d[segment_id == 2]
    x[segment_id == 2] = xmin + d[segment_id == 2] * np.cos(np.pi - theta[segment_id == 2])
    y[segment_id == 2] = ymax + d[segment_id == 2] * np.sin(np.pi - theta[segment_id == 2])

    theta[segment_id == 4] = segment_position[segment_id == 4] / d[segment_id == 4]
    x[segment_id == 4] = xmax + d[segment_id == 4] * np.cos(np.pi / 2.0 - theta[segment_id == 4])
    y[segment_id == 4] = ymax + d[segment_id == 4] * np.sin(np.pi / 2.0 - theta[segment_id == 4])

    theta[segment_id == 6] = segment_position[segment_id == 6] / d[segment_id == 6]
    x[segment_id == 6] = xmax + d[segment_id == 6] * np.cos(2.0 * np.pi - theta[segment_id == 6])
    y[segment_id == 6] = ymin + d[segment_id == 6] * np.sin(2.0 * np.pi - theta[segment_id == 6])

    theta[segment_id == 8] = segment_position[segment_id == 8] / d[segment_id == 8]
    x[segment_id == 8] = xmin + d[segment_id == 8] * np.cos(3.0 / 2.0 * np.pi - theta[segment_id == 8])
    y[segment_id == 8] = ymin + d[segment_id == 8] * np.sin(3.0 / 2.0 * np.pi - theta[segment_id == 8])

    return x, y


def make_storm_arrays_by_raincell(number_of_raincells_by_storm, storm_ids, storm_arrival_times, storm_months):
    """
    Repeat storm properties for each member raincell to help get arrays per raincell.

    """
    # Could be made more generic by taking df as argument and looping through columns...
    storm_ids_by_raincell = np.repeat(storm_ids, number_of_raincells_by_storm)
    storm_arrival_times_by_raincell = np.repeat(storm_arrival_times, number_of_raincells_by_storm)
    storm_months_by_raincell = np.repeat(storm_months, number_of_raincells_by_storm)
    return storm_ids_by_raincell, storm_arrival_times_by_raincell, storm_months_by_raincell



def construct_outer_raincells_inverse_cdf(gamma, xrange, yrange):
    """
    Empirically constructed inverse CDF of raincells in outer region.

    """
    # So that x (distance) can be looked up from (sampled) y (cdf quantile)

    # Make a sample of distances (x) corresponding with CDF quantile (y)
    y1 = np.arange(0.0, 0.01, 0.0001)
    y2 = np.arange(0.01, 0.99, 0.001)
    y3 = np.arange(0.99, 1.0+0.00001, 0.0001)
    y = np.concatenate([y1, y2, y3])
    x = []
    i = 1
    for q in y:
        r, info, ier, msg = scipy.optimize.fsolve(
            outer_raincells_cdf, 0, args=(gamma, xrange, yrange, q), full_output=True
        )
        x.append(r[0])

        # Final quantile at ~1 may be subject to convergence issues, so use previous value of x
        if ier != 1:
            if i == y.shape[0] and ier == 5:
                pass
            else:
                raise RuntimeError('Convergence error in construction of inverse CDF for outer raincells')

        i += 1

    # Construct inverse CDF function using linear interpolation
    x = np.asarray(x)
    y[-1] = 1.0
    # cdf = scipy.interpolate.interp1d(x, y)
    inverse_cdf = scipy.interpolate.interp1d(y, x)

    return inverse_cdf


def combine_inner_outer_raincells(inner_number_of_raincells_by_storm, outer_number_of_raincells_by_storm, inner_x_coords, outer_x_coords,
        inner_y_coords, outer_y_coords, inner_radii, outer_radii):
    
    number_of_raincells_by_storm = inner_number_of_raincells_by_storm + outer_number_of_raincells_by_storm
    n_storms = number_of_raincells_by_storm.shape[0]

    total_raincells = np.sum(number_of_raincells_by_storm)
    x = np.zeros(total_raincells)
    y = np.zeros(total_raincells)
    radius = np.zeros(total_raincells)

    i = 0
    inner_rc_idx = 0
    outer_rc_idx = 0
    for storm_idx in range(n_storms):
        n_inner = inner_number_of_raincells_by_storm[storm_idx]
        n_outer = outer_number_of_raincells_by_storm[storm_idx]
        for _ in range(n_inner):
            x[i] = inner_x_coords[inner_rc_idx]
            y[i] = inner_y_coords[inner_rc_idx]
            radius[i] = inner_radii[inner_rc_idx]
            inner_rc_idx += 1
            i += 1
        for _ in range(n_outer):
            x[i] = outer_x_coords[outer_rc_idx]
            y[i] = outer_y_coords[outer_rc_idx]
            radius[i] = outer_radii[outer_rc_idx]
            outer_rc_idx += 1
            i += 1

    return number_of_raincells_by_storm, x, y, radius



def find_relevant_raincells(x, y, radius, xmin, xmax, ymin, ymax):
    # Distances for raincells within y-range but outside x-range
    mask_1 = ((y >= ymin) & (y <= ymax)) & ((x < xmin) | (x > xmax))
    d1 = np.abs(x - xmin)
    d2 = np.abs(x - xmax)
    distance_1 = np.minimum(d1, d2)

    # Distances for raincells within x-range but outside y-range
    mask_2 = ((x >= xmin) & (x <= xmax)) & ((y < ymin) | (y > ymax))
    d1 = np.abs(y - ymin)
    d2 = np.abs(y - ymax)
    distance_2 = np.minimum(d1, d2)

    # Distances for raincells with x greater than xmax and y outside y-range
    mask_3 = (x > xmax) & ((y < ymin) | (y > ymax))
    d1 = ((x - xmax) ** 2 + (y - ymax) ** 2) ** 0.5
    d2 = ((x - xmax) ** 2 + (y - ymin) ** 2) ** 0.5
    distance_3 = np.minimum(d1, d2)

    # Distances for raincells with x less than xmin and y outside y-range
    mask_4 = (x < xmin) & ((y < ymin) | (y > ymax))
    d1 = ((x - xmin) ** 2 + (y - ymax) ** 2) ** 0.5
    d2 = ((x - xmin) ** 2 + (y - ymin) ** 2) ** 0.5
    distance_4 = np.minimum(d1, d2)

    # To ensure all points within domain are definitely retained
    mask_5 = ((x >= xmin) & (x <= xmax)) & ((y >= ymin) & (y <= ymax))
    distance_5 = np.zeros(mask_5.shape[0])

    # Collate minimum distances
    min_distance = np.zeros(x.shape[0])
    min_distance[mask_1] = distance_1[mask_1]
    min_distance[mask_2] = distance_2[mask_2]
    min_distance[mask_3] = distance_3[mask_3]
    min_distance[mask_4] = distance_4[mask_4]
    min_distance[mask_5] = distance_5[mask_5]

    # Identify relevant raincells (i.e. radius exceeds minimum distance to domain)
    relevant_flag = np.zeros(x.shape[0], dtype=bool)
    relevant_flag[min_distance < radius] = 1

    return relevant_flag


def outer_raincells_cdf(x, gamma, xrange, yrange, q=0):
    """
    CDF of distances of raincells in outer region according to Burton et al. (2010) equation A8.

    """
    # x = distance from domain boundaries, xrange is w and yrange is z in Burton et al. (2010)
    # returns y = cdf of distance of relevant raincells occurring in the outer region
    # additionally subtracting q (in range 0-1) to enable solving for x given a desired y
    return 1 - (1 + (4 * x * gamma) / (gamma * (xrange + yrange) + 4)) * np.exp(-gamma * x) - q




def identify_domain_bounds(grid, cell_size, points):
    """
    Set (inner) simulation domain bounds as maximum extent of output points and grid (if required).

    """
    if grid is not None:  # accounts for both catchment and grid outputs
        grid_xmin, grid_ymin, grid_xmax, grid_ymax = grid_limits(grid)
    if points is not None:
        points_xmin = np.min(points['easting'])
        points_ymin = np.min(points['northing'])
        points_xmax = np.max(points['easting'])
        points_ymax = np.max(points['northing'])
        if grid is not None:
            xmin = np.minimum(points_xmin, grid_xmin)
            ymin = np.minimum(points_ymin, grid_ymin)
            xmax = np.maximum(points_xmax, grid_xmax)
            ymax = np.maximum(points_ymax, grid_ymax)
        else:
            xmin = points_xmin
            ymin = points_ymin
            xmax = points_xmax
            ymax = points_ymax
    if cell_size is not None:
        xmin = round_down(xmin, cell_size)
        ymin = round_down(ymin, cell_size)
        xmax = round_up(xmax, cell_size)
        ymax = round_up(ymax, cell_size)
    else:
        xmin = round_down(xmin, 1)
        ymin = round_down(ymin, 1)
        xmax = round_up(xmax, 1)
        ymax = round_up(ymax, 1)
    return xmin, ymin, xmax, ymax

def round_down(x, base):
    return base * int(np.floor(x/base))

def round_up(x, base):
    return base * int(np.ceil(x/base))

def grid_limits(grid):
    xmin = grid['xllcorner']
    ymin = grid['yllcorner']
    xmax = xmin + grid['ncols'] * grid['cellsize']
    ymax = ymin + grid['nrows'] * grid['cellsize']
    return xmin, ymin, xmax, ymax


def main(spatial_model,parameters,simulation_length,month_lengths,
    season_definitions,intensity_distribution,rng,xmin,xmax,ymin,
    ymax,method,buffer_factor):
    """
    Simulates the NSRP process (point or spatial).

    Args:
        spatial_model (bool): Whether to run spatial model (True) or point model (False).
        parameters (pd.DataFrame): Parameter table from fitting stage.
        simulation_length (int): Number of years to simulate.
        month_lengths (list[int]): Number of hours in each month.
        season_definitions (dict): Mapping of month -> season.
        intensity_distribution (str): Distribution name: 'weibull', 'exponential', 'generalised_gamma'.
        rng (np.random.Generator): Random number generator.
        xmin, xmax, ymin, ymax (float): Domain bounds (m).
        method (str): Either 'buffer' or 'burton'.
        buffer_factor (float): Number of std deviations for buffer method.

    Returns:
        pd.DataFrame: DataFrame of simulated raincells with intensity, duration, arrival times, etc.
    """
    parameters = parameters.copy()

    if "month" not in parameters.columns:
        months = []
        seasons = []
        for month, season in season_definitions.items():
            months.append(month)
            seasons.append(season)
        df_seasons = pd.DataFrame({"month": months, "season": seasons})
        parameters = pd.merge(df_seasons, parameters, how="left", on="season")

    parameters.sort_values(by="month", inplace=True)
    parameters.reset_index(drop=True, inplace=True)

    if spatial_model:
        xmin /= 1000.0
        xmax /= 1000.0
        ymin /= 1000.0
        ymax /= 1000.0
        xrange = xmax - xmin
        yrange = ymax - ymin
    else:
        xrange = yrange = None  # placeholder for completeness

    
    storms, number_of_storms = simulate_storms(month_lengths, simulation_length, parameters, rng)

    if not spatial_model:
        df = simulate_raincells_point(storms, parameters, rng)
    else:
        buffer = method.lower() == "buffer"
        result = simulate_raincells_spatial(
            storms, parameters, xmin, xmax, ymin, ymax, xrange, yrange, rng, buffer, buffer_factor
        )
        # Handle one or two returned values gracefully
        if isinstance(result, tuple):
            df = result[0]
        else:
            df = result

    df = merge_parameters(df, month_lengths, simulation_length, parameters)
    df["raincell_arrival"] = df["storm_arrival"] + rng.exponential(1.0 / df["beta"])
    df["raincell_duration"] = rng.exponential(1.0 / df["eta"])
    df["raincell_end"] = df["raincell_arrival"] + df["raincell_duration"]

    if intensity_distribution == "exponential":
        df["raincell_intensity"] = rng.exponential(df["theta"])
    elif intensity_distribution == "weibull":
        df["raincell_intensity"] = scipy.stats.weibull_min.rvs(
            c=df["kappa"], scale=df["theta"], random_state=rng
        )
    elif intensity_distribution == "generalised_gamma":
        df["raincell_intensity"] = scipy.stats.gengamma.rvs(
            a=(df["kappa_1"] / df["kappa_2"]),
            c=df["kappa_2"],
            scale=df["theta"],
            random_state=rng,
        )
    else:
        raise ValueError(f"Unsupported intensity distribution: {intensity_distribution}")

    df.drop(columns=["lamda", "beta", "rho", "eta", "gamma", "theta", "kappa"],inplace=True,errors="ignore")
    return df



def simulate_raincells_point(storms, parameters, rng):
    """
    Simulate raincells for point model.  # TODO: Expand explanation

    """
    # Temporarily merging parameters here, but can be done before this method is called if generalise
    # _storm_arrays_by_raincell() method to work using all columns
    tmp = pd.merge(storms, parameters, how='left', on='month')
    tmp.sort_values(['storm_id'], inplace=True)  # checks that order matches self.storms

    # Generate Poisson random number of raincells for each storm
    number_of_raincells_by_storm = rng.poisson(tmp['nu'].values)

    # Make a master dataframe with one row per raincell along with the properties (ID, month, arrival time) of the
    # parent storm
    storm_ids_by_raincell, storm_arrivals_by_raincell, storm_months_by_raincell = make_storm_arrays_by_raincell(
        number_of_raincells_by_storm, storms['storm_id'].values, storms['storm_arrival'].values, storms['month'].values
    )
    df = pd.DataFrame({
        'storm_id': storm_ids_by_raincell,
        'storm_arrival': storm_arrivals_by_raincell,
        'month': storm_months_by_raincell,
    })

    return df


def merge_parameters(df, month_lengths, simulation_length, parameters):
    """
    Merge parameters into dataframe of all raincells.

    """
    df['month'] = lookup_months(month_lengths, simulation_length, df['storm_arrival'].values)
    parameters_subset = parameters.loc[parameters['fit_stage'] == 'final'].copy()
    # parameters_subset = parameters_subset.drop(
    #     ['fit_stage', 'converged', 'objective_function', 'iterations', 'function_evaluations'], axis=1
    # )
    parameters_subset = parameters_subset.drop([  # !221121
        'fit_stage', 'converged', 'objective_function', 'iterations', 'function_evaluations', 'delta', 'ar1_slope',
        'ar1_intercept', 'ar1_stderr'
    ], axis=1, errors='ignore')
    df = pd.merge(df, parameters_subset, how='left', on='month')
    return df

def identify_block_size(datetime_helper, season_definitions, timestep_length, discretisation_metadata,
        seed_sequence, number_of_years, spatial_model, parameters, intensity_distribution, xmin, xmax, ymin, ymax,
        output_types, points, catchments, float_precision, default_block_size, minimum_block_size,
        check_available_memory, maximum_memory_percentage, spatial_raincell_method, spatial_buffer_factor):
    
    """Identify size of blocks (number of years) needed to avoid potential memory issues in simulations."""
    block_size = min(number_of_years, default_block_size)
    found_block_size = False
    while not found_block_size:

        # Simulate a few years of NSRP process to get an idea of the memory requirements of the sampling.
        rng = np.random.default_rng(seed_sequence)
        sample_n_years = 30
        start_year = datetime_helper['year'].values[0]
        end_year = start_year + sample_n_years - 1
        dth = datetime_helper.loc[
            (datetime_helper['year'] >= start_year) & (datetime_helper['year'] <= end_year),
        ].copy()
        dth['month_id'] = np.arange(dth.shape[0])
        month_lengths = dth['n_hours'].values
        dummy1 = main(spatial_model, parameters, sample_n_years, month_lengths, season_definitions,
                      intensity_distribution, rng, xmin, xmax, ymin, ymax, spatial_raincell_method, spatial_buffer_factor)

        # Estimate memory requirements for NSRP process for length (number of years) of block
        nsrp_memory = (dummy1.memory_usage(deep=True).sum() / sample_n_years) * block_size * 1.2  # 1.2 = safety factor
        dummy1 = 0

        # Calculate memory requirements of working discretisation arrays (independent of any sampling)
        nt = int((24 / timestep_length) * 31)
        working_memory = 0
        if 'point' in output_types:
            if spatial_model:
                working_memory += int(nt * points.shape[0] * (float_precision / 8))
            else:
                working_memory += int(nt * (float_precision / 8))
        if ('catchment' in output_types) or ('grid' in output_types):
            working_memory += int(nt * discretisation_metadata[('grid', 'x')].shape[0] * (float_precision / 8))

        # Estimate memory requirements of point/catchment/grid output arrays for one block
        block_start_year = datetime_helper['year'].values[0]
        block_end_year = block_start_year + block_size - 1
        n_timesteps = datetime_helper.loc[
            (datetime_helper['year'] >= block_start_year) & (datetime_helper['year'] <= block_end_year),
            'n_timesteps'].sum()

        if spatial_model:
            if ('point' in output_types) and ('catchment' in output_types):
                n_points = points.shape[0] * catchments.shape[0]
            elif ('point' in output_types) and ('catchment' not in output_types):
                n_points = points.shape[0]
            elif ('point' not in output_types) and ('catchment' in output_types):
                n_points = catchments.shape[0]
            elif ('grid' in output_types):  
                n_points = discretisation_metadata[('grid', 'x')].shape[0]
            else:
                n_points = 1
        else:
            n_points = 1

        output_memory = int((n_timesteps * n_points) * (16 / 8))  # assuming np.float16 for output arrays only

        # Accept block size if estimated total memory is below maximum RAM percentage to use
        required_total = nsrp_memory + working_memory + output_memory
        system_total = psutil.virtual_memory().total
        estimated_percent_use = required_total / system_total * 100
        if check_available_memory:
            system_percent_available = psutil.virtual_memory().available / system_total * 100
            memory_limit = min(maximum_memory_percentage, system_percent_available)
        else:
            memory_limit = maximum_memory_percentage

        if estimated_percent_use < memory_limit:
            found_block_size = True
        if not found_block_size:
            if block_size == minimum_block_size:
                raise RuntimeError('Block size is at its minimum but memory availability appears to be insufficient.')
            else:
                new_block_size = int(np.floor(block_size / 2))
                block_size = max(new_block_size, minimum_block_size)

    return block_size

def make_datetime_helper(start_year, end_year, timestep_length, calendar):
    # Construct dataframe of core date information
    unique_years = np.arange(start_year, end_year+1)
    years = np.repeat(unique_years, 12)
    months = np.tile(np.arange(1, 12+1), unique_years.shape[0])
    leap_year = (
        ((np.mod(years, 4) == 0) & (np.mod(years, 100) == 0) & (np.mod(years, 400) == 0))
        | ((np.mod(years, 4) == 0) & (np.mod(years, 100) != 0))
    )
    days = np.tile(np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]), unique_years.shape)
    if calendar == 'gregorian':
        days[leap_year & (months == 2)] = 29
    hours = days * 24
    timesteps = hours / timestep_length
    timesteps = timesteps.astype(int)
    df = pd.DataFrame(dict(year=years, month=months, n_days=days, n_hours=hours, n_timesteps=timesteps))

    # Add helpers
    df['end_timestep'] = df['n_timesteps'].cumsum()  # beginning timestep of next month
    df['end_timestep'] = df['end_timestep'].astype(int)
    df['start_timestep'] = df['end_timestep'].shift()
    df.iloc[0, df.columns.get_loc('start_timestep')] = 0
    df['start_timestep'] = df['start_timestep'].astype(int)
    df['start_time'] = df['start_timestep'] * timestep_length
    df['end_time'] = df['end_timestep'] * timestep_length
    # df['n_hours'] = df['end_time'] - df['start_time']

    return df


def create_discretisation_metadata_arrays(points, grid, cell_size, dem):
    """
    Set up discretisation point location metadata arrays (x, y and z by point).

    """
    # Dictionary with keys as tuples of output type and metadata attribute (values as arrays)
    discretisation_metadata = {}

    # Point metadata values are arrays of length one
    if points is not None:
        discretisation_metadata[('point', 'x')] = points['easting'].values
        discretisation_metadata[('point', 'y')] = points['northing'].values
        if 'elevation' in points.columns:
            discretisation_metadata[('point', 'z')] = points['elevation'].values

    # For a grid these arrays are flattened 2D arrays so that every point has an associated x, y pair
    if grid is not None:
        x = np.arange(
            grid['xllcorner'] + cell_size / 2.0,
            grid['xllcorner'] + grid['ncols'] * cell_size,
            cell_size
        )
        y = np.arange(
            grid['yllcorner'] + cell_size / 2.0,
            grid['yllcorner'] + grid['nrows'] * cell_size,
            cell_size
        )
        y = y[::-1]  # reverse to get north-south order

        # Meshgrid then flatten gets each xy pair
        xx, yy = np.meshgrid(x, y)
        xf = xx.flatten()
        yf = yy.flatten()
        discretisation_metadata[('grid', 'x')] = xf
        discretisation_metadata[('grid', 'y')] = yf

        # Resample DEM to grid resolution (presumed coarser) if DEM present
        if dem is not None:
            dem_cell_size = dem.x.values[1] - dem.x.values[0]
            window = int(cell_size / dem_cell_size)

            # Restrict DEM to domain of output grid
            grid_xmin, grid_ymin, grid_xmax, grid_ymax = grid_limits(grid)
            mask_x = (dem.x > grid_xmin) & (dem.x < grid_xmax)
            mask_y = (dem.y > grid_ymin) & (dem.y < grid_ymax)
            dem = dem.where(mask_x & mask_y, drop=True)

            # Boundary argument required to avoid case where DEM does not match grid neatly
            resampled_dem = dem.coarsen(x=window, boundary='pad').mean(skipna=True) \
                .coarsen(y=window, boundary='pad').mean(skipna=True)
            flat_resampled_dem = resampled_dem.data.flatten()
            discretisation_metadata[('grid', 'z')] = flat_resampled_dem

    return discretisation_metadata


def simulate_realisation(
        realisation_id, datetime_helper, number_of_years, timestep_length, season_definitions,
        spatial_model, output_types, discretisation_metadata, points, catchments, parameters,
        intensity_distribution, rng, xmin, xmax, ymin, ymax, output_paths, block_size, block_subset_size,
        spatial_raincell_method, spatial_buffer_factor, simulation_mode, weather_model, max_dsl=6.0,
        n_divisions=8, do_reordering=True,
):
    """
    Simulate realisation of NSRP process and collect outputs in output_arrays.
    """
    rng2 = np.random.default_rng(rng.integers(1000000, 1000000000))

    # Initialize arrays once per realisation
    output_arrays = {}

    if simulation_mode != 'shuffling_preparation':
        discrete_rainfall = initialise_discrete_rainfall_arrays(
            spatial_model, output_types, discretisation_metadata, points, int((24 / timestep_length) * 31))

    n_blocks = int(np.ceil(number_of_years / block_size))
    block_id = 0

    while block_id * block_size < number_of_years:
        # Determine block years
        block_start_year = datetime_helper['year'].values[0] + block_id * block_size
        block_end_year = min(block_start_year + block_size - 1, datetime_helper['year'].values[-1])
        actual_block_size = block_end_year - block_start_year + 1

        # Subset datetime helper for this block
        dth = datetime_helper.loc[
            (datetime_helper['year'] >= block_start_year) &
            (datetime_helper['year'] <= block_end_year)
        ].copy()
        dth['month_id'] = np.arange(dth.shape[0])
        month_lengths = dth['n_hours'].values

        # Generate raincells
        df = main(spatial_model, parameters, actual_block_size, month_lengths, season_definitions,
                  intensity_distribution, rng, xmin, xmax, ymin, ymax, spatial_raincell_method, spatial_buffer_factor)

        # Convert km -> m
        if 'raincell_x' in df.columns:
            df['raincell_x'] *= 1000
            df['raincell_y'] *= 1000
            df['raincell_radii'] *= 1000

        if spatial_model:
            df['raincell_coverage'] = get_raincell_coverage2(
                df['raincell_x'].values, df['raincell_y'].values, df['raincell_radii'].values,
                xmin, xmax, ymin, ymax, 100, 100)
        else:
            df['raincell_coverage'] = 1.0

        # Rationalise storms and aggregate windows
        df = rationalise_storms2(dth, df, n_divisions)
        df1 = aggregate_windows(dth, df, n_divisions)

        if simulation_mode != 'shuffling_preparation':
            discretise_by_point(
                spatial_model, dth, season_definitions, df, output_types, timestep_length,
                discrete_rainfall, discretisation_metadata, points, catchments, realisation_id,
                output_paths, block_id, block_subset_size, weather_model, simulation_mode,
                output_arrays=output_arrays  # Pass the accumulating dictionary
            )

        block_id += 1

    # Write all outputs at the end of the realisation
    if simulation_mode != 'shuffling_preparation':
        write_output(output_arrays, output_paths, write_new_files=True, discretisation_metadata=discretisation_metadata)


def initialise_discrete_rainfall_arrays(spatial_model, output_types, discretisation_metadata, points, nt):
    dc = {}
    if 'point' in output_types:
        if spatial_model:
            dc['point'] = np.zeros((nt, points.shape[0]))
        else:
            dc['point'] = np.zeros((nt, 1))
    if ('catchment' in output_types) or ('grid' in output_types):
        dc['grid'] = np.zeros((nt, discretisation_metadata[('grid', 'x')].shape[0]))
    return dc

@numba.jit(nopython=True, parallel=True)
def get_raincell_coverage2(rc_xs, rc_ys, rc_rads, xmin, xmax, ymin, ymax, nx, ny):
    # assumes everything is coming in with units of metres

    domain_area = (xmax - xmin) * (ymax - ymin)

    # Initial area assuming all relevant - to be overwritten for partially overlapping cells
    areas = np.pi * rc_rads ** 2

    # Identify cells with centres inside domain
    # centre_inside = (rc_xs >= xmin) & (rc_xs <= xmax) & (rc_ys >= ymin) & (rc_ys <= ymax)

    # Identify if entirely within domain (maximum distance to domain bounds less than radius)
    #  TODO: See notes - need to check bounding x and y range
    # - x and y range etc can be calculated in a vectorised way upfront - reduce amount of stuff calculated in the loop
    rc_xmin = rc_xs - rc_rads
    rc_xmax = rc_xs + rc_rads
    rc_ymin = rc_ys - rc_rads
    rc_ymax = rc_ys + rc_rads
    inside_domain = (rc_xmin >= xmin) & (rc_xmax <= xmax) & (rc_ymin >= ymin) & (rc_ymax <= ymax)

    # Identify corners of intersecting rectangle of domain and the square that bounds the raincell circle
    sub_xmin = np.maximum(rc_xmin, xmin)
    sub_xmax = np.minimum(rc_xmax, xmax)
    sub_ymin = np.maximum(rc_ymin, ymin)
    sub_ymax = np.minimum(rc_ymax, ymax)

    # Raincell loop to check/modify raincell areas array if only partly covering domain
    # for i in range(rc_xs.shape[0]):
    for i in numba.prange(rc_xs.shape[0]):
        inside = inside_domain[i]

        if not inside:
            rc_x = rc_xs[i]
            rc_y = rc_ys[i]
            rad = rc_rads[i]
            _sub_xmin = sub_xmin[i]
            _sub_xmax = sub_xmax[i]
            _sub_ymin = sub_ymin[i]
            _sub_ymax = sub_ymax[i]

            # Identify area covered by one grid cell in the intersecting rectangle
            # - use less points in grid spacing is very small
            # nx = 100  # 1000
            # ny = 100  # 1000
            dx = (_sub_xmax - _sub_xmin) / nx  # float(nx)
            dy = (_sub_ymax - _sub_ymin) / ny  # float(ny)
            
            # Count grid cells within raincell
            grid_x = np.linspace(_sub_xmin + dx / 2.0, _sub_xmax - dx / 2.0, nx)
            grid_y = np.linspace(_sub_ymin + dy / 2.0, _sub_ymax - dy / 2.0, ny)
            # grid_count = np.sum((((_x - rc_x) ** 2.0 + (_y - rc_y) ** 2.0) ** 0.5) < rad)
            grid_count = 0
            for x in grid_x:
                for y in grid_y:
                    d = ((x - rc_x) ** 2 + (y - rc_y) ** 2) ** 0.5
                    if d < rad:
                        grid_count += 1
            areas[i] = grid_count * dx * dy

    frac_areas = areas / domain_area

    return frac_areas


def rationalise_storms2(datetime_helper, df, n_divisions):
    
    tmp = pd.concat([datetime_helper for _ in range(n_divisions)])
    tmp['window'] = np.repeat(np.arange(n_divisions), datetime_helper.shape[0])
    tmp.sort_values(['year', 'month', 'window'], inplace=True)
    tmp['window_start'] = tmp['n_hours'] / float(n_divisions) * tmp['window'] + tmp['start_time'] - tmp['start_time'].min()
    tmp['window_end'] = tmp['window_start'].shift(-1)
    tmp.iloc[-1, tmp.columns.get_loc('window_end')] = tmp['end_time'].values[-1] - tmp['start_time'].min()
    df = df.loc[df['raincell_arrival'] < tmp['window_end'].max()].copy()  # hopefully no memory issues...
    df['raincell_end'] = np.where(df['raincell_end'] > tmp['window_end'].max(), tmp['window_end'].max(), df['raincell_end'])

    # For each raincell, identify the window in which the arrival occurs
    # year_idx = np.digitize(df['storm_arrival'], period_ends, right=True)
    df['start_win_idx'] = np.digitize(df['raincell_arrival'], tmp['window_end'], right=True)

    # For each raincell, identify the window in which the end occurs
    df['end_win_idx'] = np.digitize(df['raincell_end'], tmp['window_end'], right=True)

    df['n_windows'] = df['end_win_idx'] - df['start_win_idx'] + 1
    df['duplicate_id'] = 0
    if df['n_windows'].max() > 1:
        df_subs = [df]
        for n_wins in np.unique(df['n_windows'][df['n_windows'] > 1]):
            # print(n_wins)
            for dup_id in range(1, n_wins):
                df_sub = df.loc[df['n_windows'] == n_wins].copy()
                df_sub['duplicate_id'] = dup_id
                df_subs.append(df_sub)
        df = pd.concat(df_subs)
    df.sort_values(['storm_id', 'raincell_arrival'], inplace=True)  # sort on raincell_arrival / window?

    # CHECKING - should be ok now that ensure only raincells arriving before period end are considered
    if df['n_windows'].min() < 1:
        print(df.loc[df['n_windows'] < 1])
        sys.exit()
    
    # For "duplicated" raincells, update starts/ends in line with window starts/ends
    df['start_win_idx2'] = df['start_win_idx'] + df['duplicate_id']
    df['end_win_idx2'] = df['start_win_idx2']
    df['start_win'] = tmp['window_start'].values[df['start_win_idx2']]
    df['end_win'] = tmp['window_end'].values[df['end_win_idx2']]
    df['raincell_arrival'] = np.where(df['raincell_arrival'] < df['start_win'], df['start_win'], df['raincell_arrival'])
    df['raincell_end'] = np.where(df['raincell_end'] > df['end_win'], df['end_win'], df['raincell_end'])
    df['raincell_duration'] = df['raincell_end'] - df['raincell_arrival']
    df['win_length'] = df['end_win'] - df['start_win']
    # df['raincell_depth'] = df['raincell_intensity'] * df['raincell_duration']
    df['year'] = tmp['year'].values[df['start_win_idx2']]
    df['month'] = tmp['month'].values[df['start_win_idx2']]
    df.drop(columns=[
        'start_win_idx', 'end_win_idx', 'n_windows', 'duplicate_id', 'end_win_idx2', 'end_win'],  # 'start_win',
        inplace=True
    )
    df.rename(columns={'start_win_idx2': 'win_id', 'start_win': 'win_start'}, inplace=True)

    return df  


def aggregate_windows(datetime_helper, df, n_divisions):
    # Make a datetime_helper containing the starts and ends of the windows
    # print(datetime_helper)
    tmp = pd.concat([datetime_helper for _ in range(n_divisions)])
    tmp['window'] = np.repeat(np.arange(n_divisions), datetime_helper.shape[0])
    tmp.sort_values(['year', 'month', 'window'], inplace=True)
    tmp['window_start'] = tmp['n_hours'] / float(n_divisions) * tmp['window'] + tmp['start_time'] - tmp[
        'start_time'].min()
    tmp['window_end'] = tmp['window_start'].shift(-1)
    tmp.iloc[-1, tmp.columns.get_loc('window_end')] = tmp['end_time'].values[-1] - tmp['start_time'].min()

    # Depths for windows (equivalent to storm depths but for fixed windows)
    df['raincell_depth'] = df['raincell_intensity'] * df['raincell_duration'] * df['raincell_coverage']  # !221111
    df_win = df.groupby(['win_id'])[['year', 'month', 'win_start', 'win_length', 'raincell_depth']].agg({
        'year': min, 'month': min, 'win_start': min, 'win_length': min, 'raincell_depth': sum
    })
    df_win.reset_index(inplace=True)
    df_win.rename(columns={'raincell_depth': 'win_depth'}, inplace=True)

    # Ensure serially complete
    # - assuming only window depth df (not raincell df) needs to be serially complete for now
    # print(df_win.shape[0])
    tmp1 = pd.DataFrame({
        'win_id': np.arange(tmp.shape[0]), 'year': tmp['year'], 'month': tmp['month'], 'win_start': tmp['window_start'],
        'win_length': tmp['window_end'] - tmp['window_start']}
    )
    df_win = df_win.merge(tmp1, how='outer')
    df_win['win_depth'] = np.where(~np.isfinite(df_win['win_depth']), 0.0, df_win['win_depth'])
    df_win.sort_values('win_id', inplace=True)

    return df_win



def discretise_by_point(spatial_model, datetime_helper, season_definitions, df, output_types, timestep_length,
                        discrete_rainfall, discretisation_metadata, points, catchments, realisation_id,
                        output_paths, block_id, block_subset_size, weather_model, simulation_mode,
                        output_arrays):
    """
    Accumulate outputs into output_arrays (one per realisation) over all blocks/months.
    """

    datetime_helper = datetime_helper.copy()
    initial_start_time = datetime_helper['start_time'].values[0]
    initial_start_timestep = datetime_helper['start_timestep'].values[0]
    datetime_helper['start_time'] -= initial_start_time
    datetime_helper['end_time'] -= initial_start_time
    datetime_helper['start_timestep'] -= initial_start_timestep
    datetime_helper['end_timestep'] -= initial_start_timestep

    subset_n_years = min(int(datetime_helper.shape[0] / 12), block_subset_size)
    subset_start_idx = 0

    while subset_start_idx < datetime_helper.shape[0]:
        subset_end_idx = min(subset_start_idx + subset_n_years * 12 - 1, datetime_helper.shape[0] - 1)
        subset_start_time = datetime_helper['start_time'].values[subset_start_idx]
        subset_end_time = datetime_helper['end_time'].values[subset_end_idx]
        df1 = df.loc[(df['raincell_arrival'] < subset_end_time) & (df['raincell_end'] > subset_start_time)]

        for month_idx in range(subset_start_idx, subset_end_idx + 1):
            year = datetime_helper['year'].values[month_idx]
            month = datetime_helper['month'].values[month_idx]
            season = season_definitions.get(month, -999)

            start_time = datetime_helper['start_time'].values[month_idx]
            end_time = datetime_helper['end_time'].values[month_idx]
            temporal_mask = (df1['raincell_arrival'].values < end_time) & (df1['raincell_end'].values > start_time)

            raincell_arrival_times = df1['raincell_arrival'].values[temporal_mask]
            raincell_end_times = df1['raincell_end'].values[temporal_mask]
            raincell_intensities = df1['raincell_intensity'].values[temporal_mask]

            if spatial_model:
                raincell_x = df1['raincell_x'].values[temporal_mask]
                raincell_y = df1['raincell_y'].values[temporal_mask]
                raincell_radii = df1['raincell_radii'].values[temporal_mask]

                _output_types = list(set(output_types) & set(['point', 'catchment'])) if ('catchment' in output_types and 'grid' in output_types) else output_types
                for output_type in _output_types:
                    discretisation_case = 'grid' if output_type == 'catchment' else output_type
                    discretise_spatial(
                        start_time, timestep_length, raincell_arrival_times, raincell_end_times,
                        raincell_intensities, discrete_rainfall[discretisation_case],
                        raincell_x, raincell_y, raincell_radii,
                        discretisation_metadata[(discretisation_case, 'x')],
                        discretisation_metadata[(discretisation_case, 'y')],
                        discretisation_metadata[(discretisation_case, 'phi', season)]
                    )
            else:
                discretise_point(start_time, timestep_length, raincell_arrival_times, raincell_end_times,
                                 raincell_intensities, discrete_rainfall['point'][:, 0])

            timesteps_in_month = datetime_helper.loc[
                (datetime_helper['year'] == year) & (datetime_helper['month'] == month), 'n_timesteps'
            ].values[0]

            if weather_model is not None:
                weather_model.simulate(
                    rainfall=discrete_rainfall,
                    n_timesteps=timesteps_in_month,
                    year=year,
                    month=month,
                    discretisation_metadata=discretisation_metadata,
                    output_types=output_types,
                    timestep=timestep_length
                )

            collate_output_arrays(
                output_types, spatial_model, points, catchments, realisation_id, discrete_rainfall, 'prcp',
                0, timesteps_in_month, discretisation_metadata, month_idx, output_arrays
            )

            if weather_model is not None:
                weather_model.simulator.collate_outputs(
                    output_types, spatial_model, points, catchments, realisation_id, timesteps_in_month,
                    discretisation_metadata, month_idx
                )

        subset_start_idx += subset_n_years * 12

  

@numba.jit(nopython=True)  # , parallel=True , fastmath=True
def discretise_spatial(
        period_start_time, timestep_length, raincell_arrival_times, raincell_end_times,
        raincell_intensities, discrete_rainfall,
        raincell_x_coords, raincell_y_coords, raincell_radii,
        point_eastings, point_northings, point_phi,  # point_ids,
):
    # Point loop should be parallelisable in numba, as each point has its own index in the second dimension of the
    # discrete_rainfall array

    # Modifying the discrete rainfall arrays themselves so need to ensure zeros before starting
    discrete_rainfall.fill(0.0)

    # Subset raincells based on whether they intersect the point being discretised
    for idx in range(point_eastings.shape[0]):
    # for idx in numba.prange(point_eastings.shape[0]):
        x = point_eastings[idx]
        y = point_northings[idx]
        yi = idx

        distances_from_raincell_centres = np.sqrt((x - raincell_x_coords) ** 2 + (y - raincell_y_coords) ** 2)
        spatial_mask = distances_from_raincell_centres <= raincell_radii

        discretise_point(
            period_start_time, timestep_length, raincell_arrival_times[spatial_mask],
            raincell_end_times[spatial_mask], raincell_intensities[spatial_mask], discrete_rainfall[:, yi]
        )

        discrete_rainfall[:, yi] *= point_phi[idx]


@numba.jit(nopython=True)  # , fastmath=True
def discretise_point(
        period_start_time, timestep_length, raincell_arrival_times, raincell_end_times,
        raincell_intensities, discrete_rainfall
):
    # Not parallelisable in numba as is, as could try to write to same timestep concurrently for > 1 raincell

    # Modifying the discrete rainfall arrays themselves so need to ensure zeros before starting
    discrete_rainfall.fill(0.0)

    # Discretise each raincell in turn
    for idx in range(raincell_arrival_times.shape[0]):

        # Times relative to period start
        rc_arrival_time = raincell_arrival_times[idx] - period_start_time
        rc_end_time = raincell_end_times[idx] - period_start_time
        rc_intensity = raincell_intensities[idx]

        # Timesteps relative to period start
        rc_arrival_timestep = int(np.floor(rc_arrival_time / timestep_length))
        rc_end_timestep = int(np.floor(rc_end_time / timestep_length))  # timestep containing end

        # Proportion of raincell in each relevant timestep
        for timestep in range(rc_arrival_timestep, rc_end_timestep + 1):
            timestep_start_time = timestep * timestep_length
            timestep_end_time = (timestep + 1) * timestep_length
            effective_start = np.maximum(rc_arrival_time, timestep_start_time)
            effective_end = np.minimum(rc_end_time, timestep_end_time)
            timestep_coverage = effective_end - effective_start

            if timestep < discrete_rainfall.shape[0]:
                discrete_rainfall[timestep] += rc_intensity * timestep_coverage


def collate_output_arrays(
        output_types, spatial_model, points, catchments, realisation_id, values, variable,
        timesteps_to_skip, timesteps_in_month, discretisation_metadata, month_idx, output_arrays
):
    """
    Collate simulation outputs (point, catchment, or grid) into output_arrays dictionary.
    Persistent across months and blocks.
    """
    for output_type in output_types:
        if output_type == 'point':
            location_ids = points['point_id'].values if spatial_model else [1]
        elif output_type == 'catchment':
            location_ids = catchments['id'].values
        elif output_type == 'grid':
            location_ids = [1]
        else:
            raise ValueError(f"Unknown output_type: {output_type}")

        for idx, location_id in enumerate(location_ids):
            output_key = (output_type, location_id, realisation_id, variable)

            # Select correct array slice
            if output_type == 'point':
                arr = values.get(('point', variable), list(values.values())[0])
                output_array = arr[timesteps_to_skip:timesteps_to_skip + timesteps_in_month, idx]

            elif output_type == 'catchment':
                arr = values.get(('grid', variable), values.get(('point', variable), list(values.values())[0]))
                weights = discretisation_metadata[('catchment', 'weights', location_id)]
                output_array = spatial_mean(arr, weights)[timesteps_to_skip:timesteps_to_skip + timesteps_in_month]

            elif output_type == 'grid':
                grid_x = discretisation_metadata[('grid', 'x')]
                grid_y = discretisation_metadata[('grid', 'y')]
                x_unique = np.unique(grid_x)
                y_unique = np.unique(grid_y)
                nx, ny = len(x_unique), len(y_unique)
                val_array = values.get(('grid', variable), values.get(('point', variable), list(values.values())[0]))
                grid_array = np.full((timesteps_in_month, ny, nx), np.nan, dtype=np.float16)

                for i, (px, py) in enumerate(zip(grid_x, grid_y)):
                    col = np.where(x_unique == px)[0][0]
                    row = np.where(y_unique == py)[0][0]
                    grid_array[:, row, col] = val_array[timesteps_to_skip:timesteps_to_skip + timesteps_in_month, i]

                output_array = grid_array

            # Append to existing array or create new list
            if output_key in output_arrays:
                output_arrays[output_key].append(output_array.astype(np.float16))
            else:
                output_arrays[output_key] = [output_array.astype(np.float16)]




def write_output(output_arrays, output_paths, write_new_files, discretisation_metadata, number_format=None):
    """
    Writes outputs to text files (point/catchment) or NetCDF (grid)
    """
    for output_key, array_list in output_arrays.items():
        output_type, location_id, realisation_id, variable = output_key
        output_path = output_paths[output_key]

        # Concatenate all months/blocks along time axis
        combined_array = np.concatenate(array_list, axis=0)

        # Determine formatting
        if number_format is not None:
            _fmt = number_format
        else:
            _fmt = '%.1f' if variable in ['prcp', 'tas'] else '%.2f'

        if output_type in ['point', 'catchment']:
            values = [(_fmt % v).rstrip('0').rstrip('.') for v in combined_array.flatten()]
            output_lines = '\n'.join(values)
            mode = 'w' if write_new_files else 'a'
            with open(output_path, mode) as fh:
                if not write_new_files:
                    fh.write('\n')
                fh.write(output_lines)

        elif output_type == 'grid':
            # Grid output
            grid_x = discretisation_metadata[('grid', 'x')]
            grid_y = discretisation_metadata[('grid', 'y')]
            x_unique = np.unique(grid_x)
            y_unique = np.unique(grid_y)[::-1]
            nx, ny = len(x_unique), len(y_unique)
            nt = combined_array.shape[0]

            nc = Dataset(output_path, 'w', format='NETCDF4')
            nc.createDimension('time', nt)
            nc.createDimension('y', ny)
            nc.createDimension('x', nx)

            times = nc.createVariable('time', 'f4', ('time',))
            ys = nc.createVariable('y', 'f4', ('y',))
            xs = nc.createVariable('x', 'f4', ('x',))
            var = nc.createVariable(variable, 'f4', ('time', 'y', 'x'), zlib=True)

            times[:] = np.arange(nt)
            ys[:] = y_unique
            xs[:] = x_unique
            var[:] = combined_array

            var.units = 'mm' if variable == 'prcp' else ''
            nc.close()


def spatial_mean(x, w, axis=1, w_crit=0.0):  # , w_m , y):
    y = np.average(x[:, w > w_crit], axis=axis, weights=w[w > w_crit])
    return y

def identify_unique_seasons(season_definitions):
    return list(set(season_definitions.values()))

def main_sim(spatial_model,intensity_distribution,output_types,output_folder,output_subfolders,
        output_format,season_definitions,parameters,point_metadata,catchment_metadata,grid_metadata,epsg_code,
        cell_size,dem,phi,simulation_length,number_of_realisations,timestep_length,start_year,calendar,
        random_seed,default_block_size,check_block_size,minimum_block_size,check_available_memory,
        maximum_memory_percentage,block_subset_size,project_name,spatial_raincell_method,spatial_buffer_factor,
        simulation_mode,  # 'no_shuffling', 'shuffling_preparation', 'with_shuffling'
        weather_model,max_dsl,n_divisions,do_reordering):
    # print('  - Initialising')

    # Initialisations common to both point and spatial models (derived attributes)
    realisation_ids = range(1, number_of_realisations + 1)
    output_paths = make_output_paths(
        spatial_model, output_types, output_format, output_folder, output_subfolders, point_metadata,
        catchment_metadata, realisation_ids, project_name
    )
    if random_seed is None:
        seed_sequence = np.random.SeedSequence()
    else:
        seed_sequence = np.random.SeedSequence(random_seed)

    # ---
    # Currently setting output paths as attribute of weather model not simulator, as simulator is not initialised yet
    if weather_model is not None:
        weather_model.set_output_paths(
            spatial_model, output_types, output_format, output_subfolders, point_metadata,  # output_folder,
            catchment_metadata, realisation_ids, project_name
        )
    # ---

    # Possible that 32-bit floats could be used in places, but times need to be tracked with 64-bit floats in the case
    # of long simulations for example. So fixed precision currently, as care needed if deviating from 64-bit
    float_precision = 64

    # Most of the preparation needed for simulation is only for a spatial model  # TODO: Check each case
    if spatial_model:

        # Set (inner) simulation domain bounds
        xmin, ymin, xmax, ymax = identify_domain_bounds(grid_metadata, cell_size, point_metadata)

        # Set up discretisation point location metadata arrays (x, y and z by point)
        discretisation_metadata = create_discretisation_metadata_arrays(point_metadata, grid_metadata, cell_size, dem)

        # Associate a phi value with each point
        unique_seasons = identify_unique_seasons(season_definitions)
        discretisation_metadata = get_phi(unique_seasons, dem, phi, output_types, discretisation_metadata)

        # Get weights associated with catchments for each point
        if 'catchment' in output_types:
            discretisation_metadata = get_catchment_weights(
                grid_metadata, catchment_metadata, cell_size, epsg_code, discretisation_metadata, output_types, dem,
                unique_seasons, catchment_id_field='id'
            )
            
    else:
        xmin = None
        ymin = None
        xmax = None
        ymax = None
        # discretisation_metadata = None
        discretisation_metadata = create_discretisation_metadata_arrays(point_metadata, grid_metadata, cell_size, dem)

    # !221215
    # print(discretisation_metadata[('point', 'x')].shape)
    # print(discretisation_metadata[('grid', 'x')].shape)
    # sys.exit()

    # Date/time helper - monthly time series indicating number of hours and timesteps in month
    end_year = start_year + simulation_length - 1
    datetime_helper = make_datetime_helper(start_year, end_year, timestep_length, calendar)

    # Identify block size needed to avoid memory issues
    if check_block_size:
        block_size = identify_block_size(
            datetime_helper, season_definitions, timestep_length, discretisation_metadata,
            seed_sequence, simulation_length,
            spatial_model, parameters, intensity_distribution, xmin, xmax, ymin, ymax,
            output_types, point_metadata, catchment_metadata,
            float_precision, default_block_size, minimum_block_size, check_available_memory, maximum_memory_percentage,
            spatial_raincell_method, spatial_buffer_factor
        )
    else:
        block_size = default_block_size

    # Do simulation
    rng = np.random.default_rng(seed_sequence)
    for realisation_id in realisation_ids:
        simulate_realisation(
            realisation_id, datetime_helper, simulation_length, timestep_length, season_definitions,
            spatial_model, output_types, discretisation_metadata, point_metadata, catchment_metadata,
            parameters, intensity_distribution, rng, xmin, xmax, ymin, ymax, output_paths, block_size,
            block_subset_size, spatial_raincell_method, spatial_buffer_factor, simulation_mode,
            weather_model, max_dsl, n_divisions, do_reordering,
        )

    # TODO: Implement additional output - phi, catchment weights, random seed


def get_phi(unique_seasons, dem, phi, output_types, discretisation_metadata):
    """
    Associate a phi value with each discretisation point.

    """
    # Calculate phi for each discretisation point for all output types using interpolation (unless a point is in the
    # dataframe of known phi, in which case use it directly)
    for season in unique_seasons:

        # Make interpolator (flag needed for whether phi should be log-transformed)
        if dem is not None:
            interpolator, log_transformation, significant_regression = make_phi_interpolator(
                phi.loc[phi['season'] == season]
            )
        else:
            interpolator, log_transformation, significant_regression = make_phi_interpolator(
                phi.loc[phi['season'] == season], include_elevation=True
            )

        # Estimate phi for points and grid (if phi is known at point location then exact value should be preserved)
        discretisation_types = list(set(output_types) & set(['point', 'grid']))
        if ('catchment' in output_types) and ('grid' not in output_types):
            discretisation_types.append('grid')
        for output_type in discretisation_types:
            if (dem is not None) and significant_regression:
                interpolated_phi = interpolator(
                    (discretisation_metadata[(output_type, 'x')], discretisation_metadata[(output_type, 'y')]),
                    mesh_type='unstructured',
                    ext_drift=discretisation_metadata[(output_type, 'z')],
                    return_var=False
                )
            else:
                interpolated_phi = interpolator(
                    (discretisation_metadata[(output_type, 'x')], discretisation_metadata[(output_type, 'y')]),
                    mesh_type='unstructured',
                    return_var=False
                )
            if log_transformation:
                discretisation_metadata[(output_type, 'phi', season)] = np.exp(interpolated_phi)
            else:
                discretisation_metadata[(output_type, 'phi', season)] = interpolated_phi
            discretisation_metadata[(output_type, 'phi', season)] = np.where(
                discretisation_metadata[(output_type, 'phi', season)] < 0.0,
                0.0,
                discretisation_metadata[(output_type, 'phi', season)]
            )

    return discretisation_metadata

def read_ascii_raster(file_path, data_type=float):
    # Headers
    dc = {}
    with open(file_path, 'r') as fh:
        for i in range(6):
            line = fh.readline()
            key, val = line.rstrip().split()
            key = key.lower()
            dc[key] = val

    nx = int(dc['ncols'])
    ny = int(dc['nrows'])
    cell_size = float(dc['cellsize'])
    if ('xllcorner' in dc.keys()) and ('yllcorner' in dc.keys()):
        xll = float(dc['xllcorner']) + cell_size / 2.0
        yll = float(dc['yllcorner']) + cell_size / 2.0
    elif ('xllcenter' in dc.keys()) and ('yllcenter' in dc.keys()):
        xll = float(dc['xllcenter'])
        yll = float(dc['yllcenter'])
    if data_type == float:
        nodata_flag = float(dc['nodata_value'])
    else:
        nodata_flag = int(dc['nodata_value'])

    # Values array
    arr = np.loadtxt(file_path, dtype=data_type, skiprows=6)
    arr = ma.masked_values(arr, nodata_flag)

    # Convert to xarray data array
    x = np.arange(xll, xll + cell_size * nx, cell_size)
    #y = np.arange(yll, yll + cell_size * ny, cell_size)
    y = np.linspace(yll, yll + cell_size * ny, ny, endpoint=False) #newly added on 23-Sep-2025
    y = y[::-1]  # check
    da = xr.DataArray(
        data=arr,
        dims=['y', 'x'],
        coords={
            'x': x,
            'y':  y,
        },
    )

    return da

def make_output_paths(
        spatial_model, output_types, output_format, output_folder, output_subfolders, points, catchments,
        realisation_ids, project_name, variables=['prcp'],
):
    output_paths = {}
    for output_type in output_types:
        if output_type == 'grid':
            paths = output_paths_helper(
                spatial_model, output_type, 'nc', output_folder, output_subfolders, points, catchments, realisation_ids,
                project_name, variables,
            )
        else:
            paths = output_paths_helper(
                spatial_model, output_type, output_format, output_folder, output_subfolders, points, catchments,
                realisation_ids, project_name, variables,
            )
        for key, value in paths.items():
            output_paths[key] = value
    return output_paths


def output_paths_helper(
        spatial_model, output_type, output_format, output_folder, output_subfolders, points, catchments,
        realisation_ids, project_name, variables,
):
    output_format_extensions = {'csv': '.csv', 'csvy': '.csvy', 'txt': '.txt', 'netcdf': '.nc','nc':'.nc'}

    if output_type == 'point':
        if spatial_model:
            location_ids = list(points['point_id'].values)
            location_names = list(points['name'].values)
        else:
            location_ids = [1]
            location_names = [project_name]
        output_subfolder = os.path.join(output_folder, output_subfolders['point'])
    elif output_type == 'catchment':
        location_ids = list(catchments['id'].values)
        location_names = list(catchments['name'].values)
        output_subfolder = os.path.join(output_folder, output_subfolders['catchment'])
    elif output_type == 'grid':
        location_ids = [1]
        location_names = [project_name]
        output_subfolder = os.path.join(output_folder, output_subfolders['grid'])

    output_paths = {}
    output_cases = itertools.product(realisation_ids, location_ids, variables)
    for realisation_id, location_id, variable in output_cases:
        location_name = location_names[location_ids.index(location_id)]
        output_file_name = (
            # location_name + '_r' + str(realisation_id) + '_' + variable + output_format_extensions[output_format]
                location_name + '_r' + str(realisation_id) + output_format_extensions[output_format]
        )
        output_path_key = (output_type, location_id, realisation_id, variable)
        if variable == 'prcp':
            _output_subfolder = output_subfolder
        else:
            _output_subfolder = os.path.join(output_subfolder, variable)
        output_paths[output_path_key] = os.path.join(_output_subfolder, output_file_name)

        if not os.path.exists(_output_subfolder):
            os.makedirs(_output_subfolder)

    return output_paths



def make_phi_interpolator(df1, include_elevation=True, distance_bins=7):
    """
    Make function to interpolate phi, optionally accounting for elevation dependence if significant.

    """
    # Test for elevation-dependence of phi using linear regression (trying untransformed and log-transformed phi)
    if include_elevation:
        untransformed_regression = scipy.stats.linregress(df1['elevation'], df1['phi'])
        log_transformed_regression = scipy.stats.linregress(df1['elevation'], np.log(df1['phi']))
        if (untransformed_regression.pvalue < 0.05) or (log_transformed_regression.pvalue < 0.05):
            significant_regression = True
            if untransformed_regression.rvalue >= log_transformed_regression.rvalue:
                log_transformation = False
            else:
                log_transformation = True
        else:
            significant_regression = False
            log_transformation = False
    else:
        significant_regression = False
        log_transformation = False

    # Select regression model (untransformed or log-transformed) if significant (linear) elevation dependence
    if include_elevation:
        if log_transformation:
            phi = np.log(df1['phi'])
            if significant_regression:
                regression_model = log_transformed_regression
        else:
            phi = df1['phi'].values
            if significant_regression:
                regression_model = untransformed_regression
    else:
        phi = df1['phi'].values

    # Remove elevation signal from data first to allow better variogram fit
    if include_elevation and significant_regression:
        detrended_phi = phi - (df1['elevation'] * regression_model.slope + regression_model.intercept)

    # Calculate bin edges
    if df1.shape[0] > 1:  # 061
        max_distance = np.max(scipy.spatial.distance.pdist(np.asarray(df1[['easting', 'northing']])))
    else:
        max_distance = 1.0
    max_distance = max(max_distance, 1.0)  # 060
    interval = max_distance / distance_bins
    bin_edges = np.arange(0.0, max_distance + 0.1, interval)
    bin_edges[-1] = max_distance + 0.1  # ensure that all points covered

    # Estimate empirical variogram
    if include_elevation and significant_regression:
        bin_centres, gamma, counts = gstools.vario_estimate(
            (df1['easting'].values, df1['northing'].values), detrended_phi, bin_edges, return_counts=True
        )
    else:
        bin_centres, gamma, counts = gstools.vario_estimate(
            (df1['easting'].values, df1['northing'].values), phi, bin_edges, return_counts=True
        )
    bin_centres = bin_centres[counts > 0]
    gamma = gamma[counts > 0]

    # Identify best fit from exponential and spherical covariance models
    exponential_model = gstools.Exponential(dim=2)
    _, _, exponential_r2 = exponential_model.fit_variogram(bin_centres, gamma, nugget=False, return_r2=True)
    spherical_model = gstools.Spherical(dim=2)
    _, _, spherical_r2 = spherical_model.fit_variogram(bin_centres, gamma, nugget=False, return_r2=True)
    if exponential_r2 > spherical_r2:
        covariance_model = exponential_model
    else:
        covariance_model = spherical_model

    # Instantiate appropriate kriging object
    if include_elevation and significant_regression:
        phi_interpolator = gstools.krige.ExtDrift(
            covariance_model, (df1['easting'].values, df1['northing'].values), phi, df1['elevation'].values
        )
    else:
        phi_interpolator = gstools.krige.Ordinary(
            covariance_model, (df1['easting'].values, df1['northing'].values), phi
        )

    return phi_interpolator, log_transformation, significant_regression



def get_catchment_weights(
        grid, catchments, cell_size, epsg_code, discretisation_metadata, output_types, dem,
        unique_seasons, catchment_id_field
):
    """
    Catchment weights as contribution of each (grid) point to catchment-average.

    """
    # First get weight for every point for every catchment
    # print('!!', grid)
    # sys.exit()
    grid_xmin, grid_ymin, grid_xmax, grid_ymax = grid_limits(grid)
    catchment_points = catchment_weights(
        catchments, grid_xmin, grid_ymin, grid_xmax, grid_ymax, cell_size, id_field=catchment_id_field,
        epsg_code=epsg_code
    )
    # print(catchment_points[1.0]['weight'].sum())
    # print(catchment_points[2.0]['weight'].sum())
    # print(catchment_points[3.0]['weight'].sum())
    # print(catchment_points[4.0]['weight'].sum())
    # sys.exit()
    for catchment_id, point_arrays in catchment_points.items():
        # Check that points are ordered in the same way
        assert np.min(point_arrays['x'] == discretisation_metadata[('grid', 'x')]) == 1
        assert np.min(point_arrays['y'] == discretisation_metadata[('grid', 'y')]) == 1
        # TODO: Replace checks on array equivalence with dataframe merge operation
        discretisation_metadata[('catchment', 'weights', catchment_id)] = point_arrays['weight']

    # Then rationalise grid discretisation points - if a point is not used by any catchment and grid output is not
    # required then no need to discretise it
    if ('catchment' in output_types) and ('grid' not in output_types):

        # Identify cells where any subcatchment is present (i.e. overall catchment mask)
        catchment_mask = np.zeros(discretisation_metadata[('grid', 'x')].shape[0], dtype=bool)
        for catchment_id in catchment_points.keys():
            subcatchment_mask = discretisation_metadata[('catchment', 'weights', catchment_id)] > 0.0
            catchment_mask[subcatchment_mask == 1] = 1

        # Subset static (non-seasonally varying) arrays - location, elevation and weights
        discretisation_metadata[('grid', 'x')] = discretisation_metadata[('grid', 'x')][catchment_mask]
        discretisation_metadata[('grid', 'y')] = discretisation_metadata[('grid', 'y')][catchment_mask]
        if dem is not None:
            discretisation_metadata[('grid', 'z')] = discretisation_metadata[('grid', 'z')][catchment_mask]
        for catchment_id in catchment_points.keys():
            discretisation_metadata[('catchment', 'weights', catchment_id)] = (
                discretisation_metadata[('catchment', 'weights', catchment_id)][catchment_mask]
            )

        # Subset seasonally varying arrays - phi
        for season in unique_seasons:
            discretisation_metadata[('grid', 'phi', season)] = (
                discretisation_metadata[('grid', 'phi', season)][catchment_mask]
            )

    return discretisation_metadata


def catchment_weights(
        catchment_polygons, xmin, ymin, xmax, ymax, output_grid_resolution, id_field, epsg_code,
        shapefile_grid_resolution=200
):
    # Fractional coverage of catchment in each grid cell being used in output discretisation
    # in the spatial-temporal model
    # - careful with assumptions on ordering of arrays etc

    # catchment_polygons = geopandas.read_file(catchment_polygons)
    catchment_polygons['Dummy'] = 1  # helps get fractional coverage when coarsening
    number_of_catchments = catchment_polygons.shape[0]

    discretisation_points = {}

    for index in range(number_of_catchments):

        # Duplicate the required polygon to keep the shape of the geodataframe
        # - double square brackets to make sure iloc returns a dataframe not a series
        catchment_polygon = pd.concat([catchment_polygons, catchment_polygons.iloc[[index]]])
        catchment_polygon.index = range(number_of_catchments + 1)
        catchment_polygon = catchment_polygon.loc[
            (catchment_polygon.index == index) | (catchment_polygon.index == np.max(catchment_polygon.index))
        ]
        catchment_polygon.crs = epsg_code

        # Discretisation on high resolution grid
        cube = geocube.api.core.make_geocube(
            catchment_polygon,
            measurements=["Dummy"],
            resolution=(shapefile_grid_resolution, -shapefile_grid_resolution),
            fill=0,
            geom=(
                '{"type": "Polygon", '
                + '"crs": {"properties": {"name": "EPSG:' + str(epsg_code) + '"}}, '
                + '"coordinates": [['
                + '[' + str(xmin) + ', ' + str(ymin) + '], '
                + '[' + str(xmin) + ', ' + str(ymax) + '], '
                + '[' + str(xmax) + ', ' + str(ymax) + '], '
                + '[' + str(xmax) + ', ' + str(ymin) + ']'
                + ']]'
                + '}'
            )
        )

        # Swap easting coordinates so go from low to high (east-west)
        cube = cube.reindex(x=cube.x[::-1])

        # Swap northing coordinates so go from high to low (north-south)
        cube = cube.reindex(y=cube.y[::-1])

        window = int(output_grid_resolution / shapefile_grid_resolution)
        cube = cube.coarsen(x=window).mean().coarsen(y=window).mean()

        xx, yy = np.meshgrid(cube.x, cube.y)
        xf = xx.flatten()
        yf = yy.flatten()
        weights = cube.Dummy.data.flatten()
        catchment_id = catchment_polygon.loc[catchment_polygon.index == index, id_field].values[0]
        discretisation_points[catchment_id] = {'x': xf, 'y': yf, 'weight': weights}

    return discretisation_points



############# Starting from here we see the functions put to use from preparation of time series, generation of
########## reference statistics, fitting the parameters and simulating the rainfall using the fitted parameters

timeseries_folder = '/home/users/DATA/Spatial_Inputs/Dee/GaugeData'
timeseries_format = 'csv'
SD = {12: 1, 1: 1, 2: 1,  3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 3, 9: 4, 10: 4, 11: 4}   
ALLDF = prepare_spatial_timeseries(point_metadata = pd.read_csv('/home/users/DATA/Spatial_Inputs/Dee/gauge_metadata_dee2.csv'),timeseries_folder = timeseries_folder,
                                   timeseries_format = 'csv',season_definitions = SD,completeness_threshold = 0,durations = ['1H','24H','72H','1M'],outlier_method = 'trim',maximum_relative_difference = 2,
                                   maximum_alterations = 5)


## Computation of reference statistics ##

REF = pd.concat([GetMonthStats(ALLDF[i],i,0.2) for i in np.arange(1,len(ALLDF)+1,1)],axis=0)
FINAL = pd.concat([REF,GetPooledMonthStats(ALLDF,0.2)],axis=0)
IGD = GetInterGaugeDistances('/home/users/DATA/Spatial_Inputs/Dee/gauge_metadata_dee2.csv')
CROSSCOR = pd.concat([GetCrossCorrel(ALLDF,IGD,i) for i in np.arange(1,13,1)],axis=0)
POOLCROSS = GetPoolCrossCorrel(CROSSCOR)
reference_statistics = pd.concat([FINAL,CROSSCOR.drop('distance_bin',axis=1),POOLCROSS],axis=0)
reference_statistics.loc[reference_statistics['name'] == 'cross-correlation_lag0','lag'] = 0


## Estimation of NSRP parameters ##
statistic_ids, fitting_data, ref, weights, gs = prepare(reference_statistics)
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

def fit_month_task(month):
    return fit_by_month(unique_months=[month], reference_statistics=reference_statistics,spatial_model=True,intensity_distribution='weibull',n_workers=1,            
           all_parameter_names=all_parameter_names,parameters_to_fit=parameters_to_fit,parameter_bounds=parameter_bounds,fixed_parameters=fixed_parameters,
           stage='final',initial_parameters=None,use_pooling=False)

if __name__ == '__main__':
    months = unique_months 
    with Pool() as pool:    # default uses all available CPUs
        results = pool.map(fit_month_task, months)
    # Combine the results
    parameters_df = pd.concat([res[0] for res in results], axis=0)
    fitted_stats = pd.concat([res[1] for res in results], axis=0)

# parameters_df, fitted_stats = fit_by_month(unique_months=unique_months, reference_statistics = reference_statistics, spatial_model=True, intensity_distribution='weibull', n_workers=1,
#                               all_parameter_names = all_parameter_names, parameters_to_fit= parameters_to_fit, parameter_bounds = parameter_bounds, fixed_parameters = fixed_parameters, stage='final',
#                               initial_parameters=None, use_pooling=False)

parameters_df['season'] = np.nan
parameters_df.columns = ['fit_stage', 'month', 'lamda', 'beta', 'rho', 'eta', 'gamma', 'theta','kappa', 'Converged', 'Objective_Function', 'Iterations','Function_Evaluations','intensity_distribution','season']
parameters_df.loc[parameters_df['month'].isin([12, 1, 2]), 'season'] = 1
parameters_df.loc[parameters_df['month'].isin([3, 4, 5]), 'season'] = 2
parameters_df.loc[parameters_df['month'].isin([6, 7, 8]), 'season'] = 3
parameters_df.loc[parameters_df['month'].isin([9, 10, 11]), 'season'] = 4


## Generation of rainfall realizations ##

n_realizations = 50
n_years = 30
GMETA = pd.read_csv('/home/users/DATA/Spatial_Inputs/Dee/gauge_metadata_dee2.csv')
PHIDF = pd.DataFrame({'point_id':reference_statistics['point_id'][(reference_statistics['point_id']!=-1)],'month':reference_statistics['month'][(reference_statistics['point_id']!=-1)],'phi':reference_statistics['phi'][(reference_statistics['point_id']!=-1)]})

PHIDF.index = np.arange(0,PHIDF.shape[0],1)
PHIDF = PHIDF.merge(GMETA[['point_id', 'easting', 'northing']], on='point_id', how='left')
PHIDF['season'] = np.nan
PHIDF.loc[PHIDF['month'].isin([12, 1, 2]), 'season'] = 1
PHIDF.loc[PHIDF['month'].isin([3, 4, 5]), 'season'] = 2
PHIDF.loc[PHIDF['month'].isin([6, 7, 8]), 'season'] = 3
PHIDF.loc[PHIDF['month'].isin([9, 10, 11]), 'season'] = 4

PHIDF['elevation'] = np.nan
for pid in GMETA['point_id'].unique():
    elev_value = GMETA.loc[GMETA['point_id'] == pid, 'elevation'].values[0]  
    PHIDF.loc[PHIDF['point_id'] == pid, 'elevation'] = elev_value  


rng = np.random.default_rng(seed=43)
#The DEM used here has .asc format and shall contain northing and easting, cell size in meters
dem = read_ascii_raster(file_path='/home/users/DATA/Spatial_Inputs/Dee/srtm_dee.asc', data_type=float)
# Coarsen the DEM resolution
dx = abs(float(np.diff(dem.x)[0]))
dy = abs(float(np.diff(dem.y)[0]))
target_dx = 5000*4
target_dy = 5000*4
factor_x = int(round(target_dx / dx))
factor_y = int(round(target_dy / dy))
dem_upscaled = dem.coarsen(x=factor_x, y=factor_y, boundary='trim').mean()

# Fill NaNs with random values within the range of the DEM
nan_mask = np.isnan(dem_upscaled)
random_fill = np.random.uniform(low=np.nanmin(dem_upscaled),
                                high=np.nanmax(dem_upscaled),
                                size=dem_upscaled.shape)
dem_upscaled_filled = dem_upscaled.where(~nan_mask, random_fill)

x_coords = dem_upscaled.x.values
y_coords = dem_upscaled.y.values

ncols = len(x_coords)
nrows = len(y_coords)

# Compute cell size (assumes uniform spacing)
cellsize_x = abs(np.diff(x_coords)[0])
cellsize_y = abs(np.diff(y_coords)[0])
xllcorner = x_coords.min() - cellsize_x / 2.0
yllcorner = y_coords.min() - cellsize_y / 2.0

grid_metadata = {
    "xllcorner": xllcorner,
    "yllcorner": yllcorner,
    "cellsize": float(cellsize_x),
    "ncols": ncols,
    "nrows": nrows
}


discretisation_metadata2 = create_discretisation_metadata_arrays(points = GMETA, grid = grid_metadata, cell_size = grid_metadata['cellsize'], dem = dem_upscaled)

discretisation_metadata2[('grid', 'z')] = dem_upscaled_filled.values.flatten()

DTH = make_datetime_helper(start_year = ALLDF[1]['1H']['Year'].min(), end_year = ALLDF[1]['1H']['Year'].min()+n_years-1, timestep_length = 1.0, calendar = 'gregorian')

block_size = identify_block_size( datetime_helper = DTH, season_definitions = SD, timestep_length = 1.0,
             discretisation_metadata = discretisation_metadata2, seed_sequence = 43, 
             number_of_years = n_years, spatial_model = True, parameters = parameters_df, 
             intensity_distribution = 'weibull', xmin = discretisation_metadata2[('grid', 'x')].min() ,
             xmax = discretisation_metadata2[('grid', 'x')].max() , ymin = discretisation_metadata2[('grid', 'y')].min() , 
             ymax = discretisation_metadata2[('grid', 'y')].max() , output_types = 'grid', points = GMETA.shape[0], catchments = None,
             float_precision = 64, default_block_size = 15, minimum_block_size = 1, check_available_memory = True,
             maximum_memory_percentage = 90, spatial_raincell_method = 'buffer', spatial_buffer_factor = 15 )

main_sim(spatial_model = True,intensity_distribution = 'weibull',
         output_types = ['grid'],output_folder = '/home/users/DATA/RF_Spatial_Outputs/REALIZATIONS_DEE_GRID_15KM',
         output_subfolders = {'grid': ''},output_format = 'netcdf',season_definitions = SD,parameters = parameters_df,point_metadata = GMETA,
         catchment_metadata = None,grid_metadata = grid_metadata,epsg_code = 27700, cell_size = grid_metadata['cellsize'],dem = dem_upscaled_filled,
         phi = PHIDF, simulation_length = n_years, number_of_realisations = n_realizations,timestep_length = 1.0,start_year = ALLDF[1]['1H']['Year'].min(),
         calendar = 'gregorian',random_seed = 43, default_block_size = block_size, check_block_size = True, minimum_block_size = 1, check_available_memory = True,
         maximum_memory_percentage = 90, block_subset_size = GMETA.shape[0], project_name = 'DeeDistrict', spatial_raincell_method = 'buffer',
         spatial_buffer_factor = 15, simulation_mode = 'no_shuffling', weather_model = None, max_dsl = 6, n_divisions = 8, do_reordering = True) 

# Store outputs #
os.chdir('/home/users/DATA/Spatial_Inputs/Dee/GaugeData_MOD')
for i in ALLDF.keys():
    ALLDF[i]['1H'].to_csv(GMETA['name'][i-1]+str('_MOD.csv'))

os.chdir('/home/users/DATA/Spatial_Inputs/Dee')
reference_statistics.to_csv('reference_statistics_Dee.csv',index = False)
parameters_df.to_csv('parameters_Dee.csv',index = False)
fitted_stats.to_csv('fitted_stats_Dee.csv',index = False)