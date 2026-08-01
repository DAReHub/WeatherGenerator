import os
import sys
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
import calendar



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


def transform_series(InputWeatherSeries,base_seed):
    
    df = InputWeatherSeries[0]
    df = df.rename(columns={'point_id': 'pool_id'})
    # Factors by which to stratify transformation
    #transitions = ['DDD', 'DD', 'DW', 'WD', 'WW', 'NA']  # ! is NA needed to keep serially complete here?
    variables = df['variable'].unique()
    pool_ids = df['pool_id'].unique()
    seasons = df['season'].unique()
    transformations = {}
    offsetlist = {}
    # Main loop
    dfs = []
    # for season, transition, variable, pool_id in itertools.product(self.seasons, transitions, variables, pool_ids):  # !221212
    for season, variable, pool_id in itertools.product(seasons, variables, pool_ids):  # !221212
        df1 = df.loc[
            (df['season'] == season) & (df['variable'] == variable)  # (df['transition'] == transition) &  # !221212
            & (df['pool_id'] == pool_id) & (np.isfinite(df['z_score']))
        ].copy()
        
        if (df1.shape[0] > 0) and (variable in ['temp_avg', 'dtr']):
            offset = abs(df1['z_score'].min()) + 0.01 
            bc_value, lamda = scipy.stats.boxcox(df1['z_score'] + offset)
            df1['bc_value'] = bc_value
            dfs.append(df1)
            transformations[(pool_id, variable, season, 'lamda')] = lamda
            offsetlist[(pool_id, variable, season, 'offset')] = offset

        elif (df1.shape[0] > 0) and (variable == 'sun_dur'):
            # TODO: Keep track of min/max used in scaling? Or assume min=0 and max=fully sunny day?
            # Alternatively could calculate day length here and use this - may be the most accurate

            p0 = df1.loc[df1['value'] < 0.01, 'value'].shape[0] / df1.shape[0]
            df1['scaled'] = (df1['value'] - df1['value'].min()) / (df1['value'].max() - df1['value'].min())

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                a, b, loc, scale = scipy.stats.beta.fit(df1.loc[df1['value'] >= 0.01, 'scaled'])
            transformations[(pool_id, variable, season, 'p0')] = p0
            transformations[(pool_id, variable, season, 'a')] = a
            transformations[(pool_id, variable, season, 'b')] = b
            transformations[(pool_id, variable, season, 'loc')] = loc
            transformations[(pool_id, variable, season, 'scale')] = scale

            # Recording min/max of observations for now, but see above
            transformations[(pool_id, variable, season, 'obs_min')] = df1['value'].min()
            transformations[(pool_id, variable, season, 'obs_max')] = df1['value'].max()

            # Probability associated with non-zero values
            df1['probability'] = scipy.stats.beta.cdf(df1['scaled'], a, b, loc, scale)
            df1['probability'] = (1 - p0) * df1['probability'] + p0
            df1.loc[df1['value'] < 0.01, 'probability'] = p0

            # Standard normal values - use sampling for <= p0
            rng = np.random.default_rng(seed = base_seed)  
            dummy_probability = rng.uniform(low=0, high=p0, size=df1.shape[0])
            df1['probability'] = np.where(df1['value'] < 0.01, dummy_probability, df1['probability'])
            df1['bc_value'] = scipy.stats.norm.ppf(df1['probability'], 0, 1)
            df1.drop(columns=['scaled', 'probability'], inplace=True)
            dfs.append(df1)

        elif (df1.shape[0] > 0) and (variable == 'prcp'):
            df1['bc_value'] = df1['value']
            dfs.append(df1)

        else:
            df1['bc_value'] = np.nan

    # Join all back into one dataframe
    df = pd.concat(dfs)
    df.sort_values(['pool_id', 'variable', 'datetime'], inplace=True)

    # Calculate statistics for standardisation
    df1 = df.loc[df['transition'] != 'NA']
    df1 = df1.groupby(['pool_id', 'variable', 'season'])['bc_value'].agg(['mean', 'std'])  # , 'transition'
    df1.reset_index(inplace=True)

    tmp1 = expand_grid(
        ['pool_id', 'variable', 'season'],  # , 'transition'
        df1['pool_id'].unique(), df1['variable'].unique(), df1['season'].unique()  # , df1['transition'].unique()
    )
    tmp2 = df1.groupby(['pool_id', 'variable', 'season'])[['mean', 'std']].mean()
    tmp2.reset_index(inplace=True)
    tmp2.rename(columns={'mean': 'tmp_mean', 'std': 'tmp_std'}, inplace=True)
    df1 = pd.merge(df1, tmp1, how='right')
    df1 = pd.merge(df1, tmp2, how='left')
    df1['mean'] = np.where(~np.isfinite(df1['mean']), df1['tmp_mean'], df1['mean'])
    df1['std'] = np.where(~np.isfinite(df1['std']), df1['tmp_std'], df1['std'])
    df1.drop(columns=['tmp_mean', 'tmp_std'], inplace=True)
    df1.rename(columns={'mean': 'bc_mean', 'std': 'bc_std'}, inplace=True)
    #transformed_statistics = df1  # set by returning

    # Standardise time series
    # - keep series contiguous (i.e. using NA) to ensure that lag-1 value is identified correctly
    df = pd.merge(df, df1, how='left')
    df['sd_value'] = (df['bc_value'] - df['bc_mean']) / df['bc_std']
    df['sd_lag1'] = df.groupby(['pool_id','variable', 'season'])['sd_value'].transform(shift_)

    # print(df.loc[(df['season'] == 1) & (df['variable'] == 'prcp')])  # (df['transition'] == 'DD') &
    # sys.exit()

    # Wide dataframe containing standardised values and lag-1 standardised values for all variables
    index_columns = ['pool_id', 'datetime', 'season', 'transition']  #
    tmp1 = df.pivot(index=index_columns, columns='variable', values='sd_value')
    tmp1.reset_index(inplace=True)
    tmp2 = df.pivot(index=index_columns, columns='variable', values='sd_lag1')
    tmp2.reset_index(inplace=True)
    tmp2.columns = [col + '_lag1' if col not in index_columns else col for col in tmp2.columns]
    df2 = pd.merge(tmp1, tmp2)
    return df1, df2, transformations, offsetlist
    
     
def expand_grid(column_names, *args):  # args are lists/arrays of unique values corresponding with each column
    mesh = np.meshgrid(*args)
    dc = {}
    for col, m in zip(column_names, mesh):
        dc[col] = m.flatten()
    df = pd.DataFrame(dc)
    return df

def shift_(x, lag=1):
    y = np.zeros(x.shape, dtype=x.dtype)
    y.fill(np.nan)
    y[lag:] = x[:-lag]
    return y


## The preprocessing stage for the weather variables ends here and the    
## fitting of regression model for weather variables is shown below.
## The wg.weather_model.fit fits regression models for weather variables in RWGEN
## and the function(s) for this step are given below

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



GRIDS2 = pd.read_csv('/home/users/azhar199/DATA/NEWDATA/GRIDDED_WG_QC/HOURLY_DAILY/HUKMETA2.csv')


def Get_haduk_weatherseries(NUM):
    prcp_file_path = '/home/users/azhar199/DATA/HADUK/HADUK_RF'
    tmax_file_path = '/home/users/azhar199/DATA/HADUK/HADUK_TEMPMAX'
    tmin_file_path = '/home/users/azhar199/DATA/HADUK/HADUK_TEMPMIN'
        
    RF_FILES = glob.glob(prcp_file_path+str('/*.nc'))[480:720]
    TEMP_MAX_FILES = glob.glob(tmax_file_path+str('/*.nc'))[480:720]
    TEMP_MIN_FILES = glob.glob(tmin_file_path+str('/*.nc'))[480:720]
    
    RFDS,MAXTEMP,MINTEMP = [[] for i in range(3)]
    for i in np.arange(0,len(RF_FILES),1):
        READ_RF = Dataset(RF_FILES[i])
        RF = READ_RF.variables['rainfall'][:,GRIDS2['HUK_y_index'][NUM],GRIDS2['HUK_x_index'][NUM]]
    
        READ_TMAX = Dataset(TEMP_MAX_FILES[i])
        tmax = READ_TMAX.variables['tasmax'][:,GRIDS2['HUK_y_index'][NUM],GRIDS2['HUK_x_index'][NUM]]
    
        READ_TMIN = Dataset(TEMP_MIN_FILES[i])
        tmin = READ_TMIN.variables['tasmin'][:,GRIDS2['HUK_y_index'][NUM],GRIDS2['HUK_x_index'][NUM]]
        
        RFDS.append(RF)
        MAXTEMP.append(tmax)
        MINTEMP.append(tmin)
        
    daily_ws = pd.DataFrame({'datetime':pd.date_range(start=pd.to_datetime('2001-01-01'), end=pd.to_datetime('2020-12-31'), freq="D"),
                            'prcp':np.concatenate(RFDS),'temp_max':np.concatenate(MAXTEMP),'temp_min':np.concatenate(MINTEMP)})
        
    daily_ws.index = daily_ws['datetime']
    CP1 =  pd.to_datetime(daily_ws['datetime'][0],format = '%Y-%m-%d').year
    CP2 = pd.to_datetime(daily_ws['datetime'][daily_ws.shape[0]-1],format = '%Y-%m-%d').year
    input_variables = ['temp_avg', 'dtr']
    INPUT_WEATHER_SERIES = prepare_weather_series(input_timeseries = daily_ws, input_variables = input_variables,calculation_period = [CP1,CP2], completeness_threshold = 0,
                                                  wet_threshold = 1, season_length = 'month', point_id=1)
    
    base_seed = 45
    TRANS_SERIES =  transform_series(InputWeatherSeries = INPUT_WEATHER_SERIES, base_seed = base_seed)
    TRANS_SERIES[0].rename(columns={'bc_mean': 'mean','bc_std': 'std'}, inplace=True)
    KEYS_off = TRANS_SERIES[3].keys()
    OFFSET_DF = pd.concat([pd.DataFrame([{'variable':list(KEYS_off)[i][1],'season':list(KEYS_off)[i][2],
                                          'offset':TRANS_SERIES[3][list(KEYS_off)[i]]}]) for i in np.arange(0,len(KEYS_off),1)],axis=0).reset_index(drop=True)
    
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
    
    REGRESSED_SERIES = do_regression(TRANSFORMED_SERIES = TRANS_SERIES,input_variables = input_variables)

    df_combined = pd.concat([t[4] for t in REGRESSED_SERIES[0]], ignore_index=True) 
    df_combined = df_combined.sort_values(by='datetime').drop_duplicates().reset_index(drop=True)
    
    # parameters from do_regression()
    rows = []
    for (site_id, month, variable, transition), coeffs in REGRESSED_SERIES[1].items():
        row = {"site_id": site_id,"month": month,"variable": variable,"transition": transition,}
        for i, val in enumerate(coeffs, start=1):
            row[f"beta{i}"] = val
        rows.append(row)
    
    WG_parameters = pd.DataFrame(rows)
    WG_parameters.sort_values(["site_id", "month", "variable", "transition"], inplace=True)
    WG_parameters.reset_index(drop=True, inplace=True)
    
    # residuals from do_regression
    KEYS = REGRESSED_SERIES[2].keys()
    res_df = []
    for i in np.arange(0,len(KEYS),1):
        key = list(KEYS)[i]
        trail = REGRESSED_SERIES[2][key]
        trail['variable'] = key[2]
        trail.rename(columns={trail.columns[2]: 'value'}, inplace=True)
        res_df.append(trail)
        
    RES_DF = pd.concat(res_df,axis=0)
    RES_DF = RES_DF.sort_values(by='datetime').drop_duplicates().reset_index(drop=True)

    COR_DF = (pd.DataFrame([(k[0], k[1], k[2], k[3], v) for k, v in REGRESSED_SERIES[3].items()],columns=['pool_id', 'season', 'variable', 'transition', 'r2']).sort_values(['pool_id', 'season', 'variable', 'transition']).reset_index(drop=True))
    
    # standard errors from do_regression()
    rows2 = []
    for key, value in REGRESSED_SERIES[4].items():
        site_id, month, variable, transition = key
        rows2.append({"site_id": site_id,"month": month,"variable": variable,"transition": transition,"value": value})
    
    standard_error = pd.DataFrame(rows2)
    standard_error.sort_values(["site_id", "month", "variable", "transition"], inplace=True)
    standard_error.reset_index(drop=True, inplace=True)
    
    transformed_statistics_dict = {
        (int(row['pool_id']), row['variable'], int(row['season'])): (float(row['mean']), float(row['std']))
        for _, row in TRANS_SERIES[0].iterrows()
    }
    
    transformations_fixed = {
        (int(k[0]), k[1], int(k[2]), k[3]): v
        for k, v in TRANS_SERIES[2].items()
    }

    transformed_statistics_df = pd.DataFrame([{'pool_id': pool_id,'variable': variable,'season': season,'mean': mean,'std': std}
    for (pool_id, variable, season), (mean, std) in transformed_statistics_dict.items()])

    transformed_statistics_df = transformed_statistics_df.sort_values(['pool_id', 'season', 'variable']).reset_index(drop=True)

    transformations_fixed_df = pd.DataFrame([{'pool_id': pool_id,'variable': variable,'season': season,'parameter': parameter,'value': value}
    for (pool_id, variable, season, parameter), value in transformations_fixed.items()])

    transformations_fixed_df = transformations_fixed_df.sort_values(['pool_id', 'season', 'variable', 'parameter']).reset_index(drop=True)

    northing = GRIDS2['northing'][NUM]
    easting = GRIDS2['easting'][NUM]

    dfs = [INPUT_WEATHER_SERIES[1],WG_parameters,COR_DF,standard_error,transformations_fixed_df,transformed_statistics_df,OFFSET_DF,RES_DF,]

    for df in dfs:
        df['easting'] = easting
        df['northing'] = northing
        
    return INPUT_WEATHER_SERIES[1], WG_parameters, COR_DF, standard_error, transformations_fixed_df, transformed_statistics_df, OFFSET_DF, RES_DF


if __name__ == "__main__":
    NUM = int(sys.argv[1])
    outdir = "/home/users/azhar199/DATA/NEWDATA/GRIDDED_WG_QC/HOURLY_DAILY/PARAM5KM/WGENPARAMS"
    os.makedirs(outdir,exist_ok=True)
    IWS,WGPAR,CORDF,SER,TFD,TSD,ODF,RESDF = Get_haduk_weatherseries(NUM)
    IWS.to_parquet(os.path.join(outdir, f"GRID_{NUM}_IWS.parquet"),index=False)
    WGPAR.to_parquet(os.path.join(outdir, f"GRID_{NUM}_WGPAR.parquet"),index=False)
    CORDF.to_parquet(os.path.join(outdir, f"GRID_{NUM}_CORDF.parquet"),index=False)
    SER.to_parquet(os.path.join(outdir, f"GRID_{NUM}_SER.parquet"),index=False)
    TFD.to_parquet(os.path.join(outdir, f"GRID_{NUM}_TFD.parquet"),index=False)
    TSD.to_parquet(os.path.join(outdir, f"GRID_{NUM}_TSD.parquet"),index=False)
    ODF.to_parquet(os.path.join(outdir, f"GRID_{NUM}_ODF.parquet"),index=False)
    RESDF.to_parquet(os.path.join(outdir, f"GRID_{NUM}_RESDF.parquet"),index=False)
