import pandas as pd
import numpy as np
import scipy as sp
import re
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.base import BaseEstimator, TransformerMixin
from workalendar.usa import UnitedStates


def reduce_mem_usage(df, verbose=True):
    '''
    Function to reduce size of a dataframe.
    Original code by @gemartin
    https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
    '''
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024 ** 2

    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if (c_min > np.iinfo(np.int8).min and
                        c_max < np.iinfo(np.int8).max):
                    df[col] = df[col].astype(np.int8)
                elif (c_min > np.iinfo(np.int16).min and
                      c_max < np.iinfo(np.int16).max):
                    df[col] = df[col].astype(np.int16)
                elif (c_min > np.iinfo(np.int32).min and
                      c_max < np.iinfo(np.int32).max):
                    df[col] = df[col].astype(np.int32)
                elif (c_min > np.iinfo(np.int64).min and
                      c_max < np.iinfo(np.int64).max):
                    df[col] = df[col].astype(np.int64)
            else:
                if (c_min > np.finfo(np.float16).min and
                        c_max < np.finfo(np.float16).max):
                    df[col] = df[col].astype(np.float16)
                elif (c_min > np.finfo(np.float32).min and
                      c_max < np.finfo(np.float32).max):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024 ** 2

    if verbose:
        mem_msg = 'Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'
        print(mem_msg.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df


def load_data(type, path):
    '''
    Load relevant data from raw csv files for either the train or test case.
    `type` must be one of ['train', 'test']
    Loads 3 files:
      1. train/test data
      2. building data
      3. weather data for train/test set
    Compresses files for numeric columns.
    '''
    if type == 'train':
        target_data_path = path + '/train.csv'
        weather_path = path + '/weather_train.csv'
    elif type == 'test':
        target_data_path = path + '/test.csv'
        weather_path = path + '/weather_test.csv'
    else:
        print('`type` must be either `train` or `test`')
        return(None)

    target_data = pd.read_csv(target_data_path,
                              dtype={'building_id': np.uint16,
                                     'meter_reading': np.float32})

    building = pd.read_csv(path + '/building_metadata.csv',
                           dtype={'building_id': np.uint16,
                                  'site_id': np.uint8})

    weather = pd.read_csv(weather_path,
                          dtype={'site_id': np.uint8,
                                 'air_temperature': np.float16,
                                 'year_built': np.uint8,
                                 'cloud_coverage': np.float16,
                                 'dew_temperature': np.float16,
                                 'precip_depth_1_hr': np.float16})

    # Reduce memory usage
    target_data = reduce_mem_usage(target_data)
    building = reduce_mem_usage(building)
    weather = reduce_mem_usage(weather)

    # Parse timestamps
    target_data['timestamp'] = pd.to_datetime(target_data['timestamp'],
                                              errors='coerce')
    weather['timestamp'] = pd.to_datetime(weather['timestamp'],
                                          errors='coerce')

    ''' Change string objects to categorical dtype to save memory'''
    # Convert `meter` to categorical
    meter_types = ['electricity', 'chilledwater', 'steam', 'hotwater']
    meter_types_dict = dict(zip(range(len(meter_types)), meter_types))

    target_data['meter'] = target_data['meter'].map(meter_types_dict)

    target_data['meter'] = pd.Categorical(target_data['meter'],
                                          categories=meter_types,
                                          ordered=False)
    # Convert `primary_use` to categorical
    primary_use_list = sorted(building.primary_use.unique())
    building['primary_use'] = pd.Categorical(building['primary_use'],
                                             categories=primary_use_list,
                                             ordered=False)

    return(target_data, building, weather)


def add_time_segments(df, ts_col):
    '''
    Add month, dayofweek, hour to dataframe based on `ts_col`.
    Any row that does not return a valid datetime will be set to NaN.
    '''
    d = df.copy()
    d[ts_col] = pd.to_datetime(d[ts_col], errors='coerce')
    d['month'] = d[ts_col].dt.month
    d['dayofweek'] = d[ts_col].dt.dayofweek
    d['hour'] = d[ts_col].dt.hour
    return(d)


def find_peak_temp(weather):
    '''
    Find average peak temperature by `site_id` in weather dataframe
    in order to localize timestamps.
    Assume average peak temperature should be at 2pm (14).
    Return dataframe which maps site_id to hour-offsets based on peak temps.
    '''
    weather_cp = weather.copy().drop(columns=['hour_offset'], errors='ignore')

    # Find average temperature by hour
    pks = (weather_cp.assign(hour=lambda df: df['timestamp'].dt.hour)
           .groupby(['site_id', 'hour'])
           .agg({'air_temperature': 'mean'})
           .reset_index())

    # Find hour per site that has highest average temperature
    pks_ind = (pks.groupby('site_id')['air_temperature']
               .transform(max) == pks['air_temperature'])
    site_max = (pks.loc[pks_ind, ['site_id', 'hour']]
                .rename(columns={'hour': 'hour_max_temp'}))

    # Assign hour offset
    site_max = site_max.assign(hour_offset=lambda df: df.hour_max_temp - 14)

    return(site_max)


def align_timestamp_local(df, site_max):
    '''
    Using site_max (map of site_id to hour-offsets based on peak temperature),
    add apply offset to the df
    '''
    df_cp = df.copy().drop(columns=['hour_offset'], errors='ignore')

    # Add offset back to weather df
    df_cp = df_cp.merge(site_max[['site_id', 'hour_offset']],
                        how='left', on='site_id')

    # Apply hour offset to every row in dataframe
    df_cp['timestamp_local'] = df_cp.timestamp - \
        pd.to_timedelta(df_cp.hour_offset, unit='H')

    return(df_cp)


def transform_cyclic_var(df, col, total_time, plot=False):
    '''
    Transform cyclic variables into sin/cos transformations.
    '''

    sin_col = col + '_sin'
    cos_col = col + '_cos'

    t = df.copy()
    t[sin_col] = np.sin(2 * np.pi * df[col] / total_time)
    t[cos_col] = np.cos(2 * np.pi * df[col] / total_time)

    if plot:
        plot_data = t.sort_values([col]).sample(1000)
        plt.scatter(x=plot_data[sin_col], y=plot_data[cos_col], c=None)
        plt.title('Sin-Cos Transformation of {} Var'.format(col))
        plt.show()

    return(t)


def mahalanobis(x=None, data=None, cov=None):
    """
    Compute the Mahalanobis Distance between each row of x and the data
    x    : vector or matrix of data with, say, p columns.
    data : ndarray of the distribution from which Mahal-distance of each
           observation of x is to be computed.
    cov  : covariance matrix (p x p) of the distribution.
           If None, will be computed from data.

    Note: All rows in x & data must be numeric.
          The covariance matrix must be invertible.
    """
    x_minus_mu = x - np.mean(data)

    if not cov:
        cov = np.cov(data.values.T)

    inv_covmat = sp.linalg.inv(cov)
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)

    return mahal.diagonal()


def fill_missing_time(df, groupby):
    '''
    Create all possible timestamps in dataframe for each group.
    Adds a column `filled_in` to identify original dataframe columns.
    '''

    # Make sure groupby & `timestamp` columns are present
    if len({groupby, 'timestamp'}.intersection(set(df.columns))) != 2:
        print('Error: the groupby or `timestamp` columns not supplied in df')
        return(None)

    groups = pd.DataFrame(df[groupby].unique(),
                          columns=[groupby]).assign(join_id=1)

    # Create all possible timestamp-hours
    min_ts = str(df.timestamp.dt.date.min())
    max_ts = str(df.timestamp.dt.date.max() + timedelta(days=1))
    hours = pd.DataFrame(pd.date_range(start=min_ts, end=max_ts, freq='H',
                                       closed='left'),
                         columns=['timestamp']).assign(join_id=1)

    # Create all possible site-id & timestamp-hour combinations
    all_times = groups.merge(hours, on='join_id')

    # Left join with original data, keep track of original rows
    all_times = (all_times
                 .merge(df.assign(filled_in=1), how='left',
                        on=[groupby, 'timestamp'])
                 .sort_values([groupby, 'timestamp'])
                 .drop(columns=['join_id']))

    return(all_times)


def ffill_weather(df):
    '''
    Forward-fill missing data in weather dataframes.
    For observations where the first value is null,
    fill with the average between sites of first timestamp values.
    '''
    cols_to_fill = ['air_temperature', 'cloud_coverage', 'dew_temperature',
                    'precip_depth_1_hr', 'sea_level_pressure',
                    'wind_direction', 'wind_speed']

    df_copy = df.copy()

    # Fill in missing timestamps
    df_copy = fill_missing_time(df, 'site_id')

    # Set index
    df_copy = df_copy.sort_values('timestamp').set_index('timestamp')

    # Forward fill by site_id
    df_copy[cols_to_fill] = (df_copy
                             .groupby('site_id', as_index=False)[cols_to_fill]
                             .ffill())

    # Any leftover NA's should be handled by averaging first entry by site
    min_idx = str(df_copy.index.min())
    to_fill = df_copy.loc[min_idx][cols_to_fill].copy()
    to_fill = to_fill.mean()

    # Fill precip depth with 0 if NA
    if np.isnan(to_fill['precip_depth_1_hr']):
        to_fill['precip_depth_1_hr'] = 0

    # Fill in last NA's
    for i in cols_to_fill:
        df_copy.loc[df_copy[i].isnull(), i] = to_fill[i]

    df_copy = (df_copy#.loc[df_copy['filled_in'] == 1]
               .drop(columns=['filled_in'])
               .reset_index())

    return(df_copy)


def calculate_lag(df, groupby, lag_vars, max_lag_step=3):
    '''
    Calculate specified lag variables on supplied data by
    `site_id` & hourly `timestamp`.
    Computes lags up to 5 timesteps.
    '''

    # Create all possible site_id's with a column to join
    all_times = fill_missing_time(df, groupby)

    # Calculate lags on specific variables
    for l in range(1, max_lag_step + 1):
        lagged_data = all_times.groupby(groupby).shift(l)[lag_vars]
        lagged_data = lagged_data.rename(
            columns=dict(zip(df.columns, df.columns + '_lag_' + str(l))))
        all_times = pd.concat([all_times, lagged_data], axis=1)

    res = (all_times#.loc[all_times.filled_in == 1]
           .drop(columns=['filled_in']))

    return(res)


def combine_data(target_data, building, weather, site_max_temp,
                 lag_vars=['air_temperature', 'dew_temperature',
                           'sea_level_pressure'],
                 max_lag_step=3):
    '''
    * Combine dataframes into one dataframe with all variables.
    * Local timestamps aligned through `site_max_temp`,
      which must be based only on training data.
    * Compute lag data here, since it is more efficient than computing
      on the combined set after joining.
    '''

    target_data_c = target_data.copy()
    building_c = building.copy()
    weather_c = weather.copy()

    # Fill in NA's in weather data
    weather_c = ffill_weather(weather_c)

    # Create lag data
    weather_c = calculate_lag(weather_c, 'site_id', lag_vars=lag_vars,
                              max_lag_step=max_lag_step)

    # Merge data
    target_data_c = target_data_c.merge(building_c, on='building_id',
                                        how='left')
    target_data_c = target_data_c.merge(weather_c, on=['site_id', 'timestamp'],
                                        how='left')

    # Add localized timestamp & time sigments
    target_data_c = align_timestamp_local(target_data_c, site_max_temp)
    target_data_c = add_time_segments(target_data_c, 'timestamp_local')

    return(target_data_c)


def tag_training_problems(train_df):
    '''
    Tag problem meter-reading data in training set.
    Tag conditions come from EDA.
    '''
    df = train_df.copy()

    df['problem_tag'] = 0
    df.loc[(df['building_id'].isin([1099, 1197])) & (df['meter'] == 'steam'),
           'problem_tag'] = 1
    df.loc[(df['building_id'].isin([778])) & (df['meter'] == 'chilledwater'),
           'problem_tag'] = 1
    df.loc[(df['building_id'].isin([1021])) & (df['meter'] == 'hotwater'),
           'problem_tag'] = 1

    problem_perc = round(df['problem_tag'].sum() / len(df) * 100, 2)

    print('{}% of meter-reading tagged as problem.'.format(problem_perc))
    return(df)


class BasePreprocessor(BaseEstimator, TransformerMixin):
    '''
    A class to apply preprocessing base transformations to the data.

    Options to apply transformations:
        1. building_age (from year_built)
        2. local_time_segments (get month/dayofweek/hour)
        3. cyclic_transform_local_time (sin-cos transf of time segments)
        4. cyclic_transform_wind_direction (sin-cos transf of wind_direction)
        5. log-transform square_feet
        6. square-root-transform wind_speed
        7. flag holidays
        8. flag weekends

    This will drop original columns after transformations.
    '''

    def __init__(self,
                 bin_precip=True,
                 building_age=True,
                 local_time_segments=True,
                 cyclic_transform_local_time=True,
                 cyclic_transform_wind_direction=True,
                 log_square_feet=True,
                 sqrt_wind_speed=True,
                 flag_holiday=True,
                 flag_weekend=True):

        self.drop_features = ['meter', 'meter_reading', 'meter_reading_log1p',
                              'row_id', 'timestamp', 'timestamp_local',
                              'hour_offset']
        self.bin_precip = bin_precip
        self.building_age = building_age
        self.local_time_segments = local_time_segments
        self.cyclic_transform_local_time = cyclic_transform_local_time
        self.cyclic_transform_wind_direction = cyclic_transform_wind_direction
        self.log_square_feet = log_square_feet
        self.sqrt_wind_speed = sqrt_wind_speed
        self.flag_holiday = flag_holiday
        self.flag_weekend = flag_weekend
        self.rename_features = {}

    def fit(self, X, y=None):
        return self

    def _bin_precip(self, X):
        # Bin precip_depth_1_hr, account for -1 values
        col = 'precip_depth_1_hr'
        bin_col = col + '_bin'

        bin_levels = ['-1 error', '0 mm', '1-19 mm', '20-99 mm', '100+ mm',
                      'unknown']

        X[bin_col] = X[col]
        X.loc[X[col] == -1, bin_col] = '-1 error'
        X.loc[X[col] == 0, bin_col] = '0 mm'
        X.loc[(X[col] >= 1) & (X[col] < 20), bin_col] = '1-19 mm'
        X.loc[(X[col] >= 20) & (X[col] < 100), bin_col] = '20-99 mm'
        X.loc[X[col] >= 100, bin_col] = '100+ mm'
        X.loc[X[col].isnull(), bin_col] = 'unknown'

        X[bin_col] = pd.Categorical(X[bin_col],
                                    categories=bin_levels,
                                    ordered=False)

        self.drop_features.append('precip_depth_1_hr')
        return(X)

    def _get_building_age(self, X):
        X['building_age'] = 2019 - X['year_built']
        self.drop_features.append('year_built')
        return(X)

    def _get_local_time_segments(self, X):
        X = add_time_segments(X, 'timestamp_local')
        return(X)

    def _cyclic_transform_local_time(self, X):
        X = transform_cyclic_var(X, 'month', 12)
        X = transform_cyclic_var(X, 'dayofweek', 7)
        X = transform_cyclic_var(X, 'hour', 24)
        self.drop_features += ['month', 'dayofweek', 'hour']
        return(X)

    def _cyclic_transform_wind_direction(self, X):
        X = transform_cyclic_var(X, 'wind_direction', 360)
        self.drop_features.append('wind_direction')
        return(X)

    def _get_log_square_feet(self, X):
        X['square_feet'] = np.log1p(X['square_feet'])
        self.rename_features['square_feet'] = 'square_feet_log1p'
        return(X)

    def _get_sqrt_wind_speed(self, X):
        X['wind_speed'] = np.sqrt(X['wind_speed'])
        self.rename_features['wind_speed'] = 'wind_speed_sqrt'
        return(X)

    def _flag_holiday(self, X):
        # Identify holidays in USA
        cal = UnitedStates()
        us_hols = []

        for y in [2016, 2017, 2018]:
            hol_year = cal.holidays(y)
            hol_year = [i[0] for i in hol_year]
            us_hols += hol_year

        if 'timestamp_local' in X.columns.tolist():
            ts_col = 'timestamp_local'
        else:
            ts_col = 'timestamp'

        X['is_holiday'] = X[ts_col].dt.date.isin(us_hols) * 1

        return(X)

    def _flag_weekend(self, X):
        X_c = X.copy()

        if 'timestamp_local' in X.columns.tolist():
            ts_col = 'timestamp_local'
        else:
            ts_col = 'timestamp'

        if 'dayofweek' not in X.columns.tolist():
            X_c[ts_col] = pd.to_datetime(X_c[ts_col], errors='coerce')
            X_c['dayofweek'] = X[ts_col].dt.dayofweek

        dayofweek_map = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 1}
        X['is_weekend'] = X_c['dayofweek'].map(dayofweek_map)

        return(X)

    def transform(self, X, y=None):
        X_tf = X.copy()

        if self.bin_precip:
            X_tf = self._bin_precip(X_tf)
        if self.building_age:
            X_tf = self._get_building_age(X_tf)
        if self.local_time_segments:
            X_tf = self._get_local_time_segments(X_tf)
        if self.cyclic_transform_local_time:
            X_tf = self._cyclic_transform_local_time(X_tf)
        if self.cyclic_transform_wind_direction:
            X_tf = self._cyclic_transform_wind_direction(X_tf)
        if self.log_square_feet:
            X_tf = self._get_log_square_feet(X_tf)
        if self.sqrt_wind_speed:
            X_tf = self._get_sqrt_wind_speed(X_tf)
        if self.flag_holiday:
            X_tf = self._flag_holiday(X_tf)
        if self.flag_weekend:
            X_tf = self._flag_weekend(X_tf)

        # Drop unnecessary columns
        X_tf = X_tf.drop(columns=self.drop_features, axis=1, errors='ignore')

        # Rename columns
        X_tf = X_tf.rename(columns=self.rename_features)

        # Reduce memory
        X_tf = reduce_mem_usage(X_tf, verbose=False)

        return(X_tf)


class FeatureSelector(BaseEstimator, TransformerMixin):
    '''
    A class to select feature names for further transformations:

    - categorical:        Categorical variables to CatBoost encode
    - categorical_cyclic: Categorical variables transformed via sin-cos
    - numeric:            Numeric variables, includes binary
    '''

    def __init__(self, feature_type):
        self.feature_type = feature_type
        self.cat_cols_w_num_dtype = ['building_id', 'site_id', 'month',
                                     'dayofweek', 'hour']
        self.num_cols = ['air_temperature', 'dew_temperature',
                         'cloud_coverage', 'sea_level_pressure',
                         'wind_speed', 'wind_direction', 'wind_speed_sqrt',
                         'square_feet', 'square_feet_log1p',
                         'year_built', 'building_age', 'floor_count',
                         'is_weekend', 'is_holiday']
        self.drop_columns = ['index', 'meter_reading', 'meter_reading_log1p']

    def fit(self, X, y=None):
        return(self)

    def transform(self, X, y=None):
        '''
        Return appropriate columns of X.
        '''

        # Get colnames & dtypes
        all_cols = X.columns.tolist()
        all_col_dtypes = [str(X[c].dtype) for c in all_cols]

        cat_names = []
        cat_cyclic_names = []
        numeric_names = []

        for i in range(len(all_cols)):
            colname = all_cols[i]
            dtype = all_col_dtypes[i]

            if colname[-4:] == '_sin' or colname[-4:] == '_cos':
                cat_cyclic_names.append(colname)
            elif dtype == 'object' or dtype == 'category':
                cat_names.append(colname)
            elif dtype != 'datetime64[ns]':
                # we don't need datetime objects, these are other numeric cols
                if colname in self.cat_cols_w_num_dtype:
                    cat_names.append(colname)
                elif colname in self.num_cols:
                    numeric_names.append(colname)
                elif re.search('_lag_', colname):
                    numeric_names.append(colname)

        if self.feature_type == 'categorical':
            feature_names = cat_names
        elif self.feature_type == 'categorical_cyclic':
            feature_names = cat_cyclic_names
        elif self.feature_type == 'numeric':
            feature_names = numeric_names
        else:
            print('''Error - FeatureSelector `feature_type` must be either
                  `categorical`, `categorical_cyclic`, or `numeric`.''')
            return(None)

        X_tf = X[feature_names].copy()
        X_tf = X_tf.reset_index().drop(columns=self.drop_columns,
                                       errors='ignore')

        return(X_tf)


class CategoricalTransformer(BaseEstimator, TransformerMixin):
    '''
    Class to transform all numeric columns to category columns.
    Used in categorical pipeline to ensure CatBoost encoder encodes all columns
      without needing colnames explicitly passed in.
    '''

    def __init__(self):
        self.cat_cols_w_num_dtype = ['building_id', 'site_id', 'month',
                                     'dayofweek', 'hour']

    def fit(self, X, y=None):
        return(self)

    def transform(self, X, y=None):
        X_tf = X.copy()

        all_cols = X_tf.columns
        all_col_dtypes = [str(X_tf[c].dtype) for c in all_cols]

        for i in range(len(all_cols)):
            col = all_cols[i]
            dtype = all_col_dtypes[i]

            if dtype != 'object' and dtype != 'category':
                if col in self.cat_cols_w_num_dtype:
                    X_tf[col] = X_tf[col].astype('object')

        return(X_tf)
