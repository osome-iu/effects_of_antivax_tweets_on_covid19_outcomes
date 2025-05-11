import pandas as pd
import numpy as np

from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA



#general feature engineering

def get_per_capita(df, cols, drop=False):
    for col in cols:
        df[col+'_per_capita'] = df[col]/df['population']
        if drop: df = df.drop(columns = [col])
    return df

def get_tlags(df, cols, lags = [1], drop=False, suffix=''): #lags in units of days
    lagged_dfs = list()
    for lag in lags:
        lag_col_renames = dict(zip(cols, [x+'_tlag'+str(lag)+suffix for x in cols]))
        df_lagged = df.copy()[['date','FIPS']+cols].rename(columns = lag_col_renames)
        df_lagged['date'] = df_lagged['date']+pd.DateOffset(days=lag)
        lagged_dfs.append(df_lagged)
    for lagged_df in lagged_dfs:
        df = df.merge(lagged_df, on=['date','FIPS'], how='inner')
    if drop: df = df.drop(columns = cols) 
    return df

def get_dxdts(df, cols, lags = [1], drop=False):  #lags in units of days
    df = get_tlags(df, cols, lags, drop=False, suffix='_temp')
    for lag in lags:
        temp_lag_cols = [col+'_tlag'+str(lag)+'_temp' for col in cols]
        for i,_ in enumerate(cols):
            df[cols[i]+'_dxdt'+str(lag)] = (df[cols[i]] - df[temp_lag_cols[i]])/lag
    all_temp_lag_cols = [col+'_tlag'+str(lag)+'_temp' for col in cols for lag in lags]
    df = df.drop(columns = all_temp_lag_cols)
    if drop: df = df.drop(columns = cols)
    return df

def weighted_mean(data, weights):
    x = np.array(data)
    weights = np.array(weights)/np.sum(weights)
    assert len(x)==len(weights)
    weighted_mean = np.average(x, weights=weights)
    return weighted_mean

def is_monotonic(x):
    current_greater_than_previous = list()
    for i in range(1,len(x)):
        current_greater_than_previous.append(x[i]>=x[i-1])
    return all(current_greater_than_previous)

def make_monotonic(x, epsilon=0, prioritize_last=True):
    if prioritize_last:
        for i in range(len(x)-2,-1,-1): #reversed
            x[i] = (x[i] if (x[i] < x[i+1]) else (x[i+1]-epsilon))
    else:
        for i in range(1,len(x)):
            x[i] = (x[i] if (x[i] > x[i-1]) else (x[i-1]+epsilon))
    return x

def make_df_monotonic(df, cols, epsilon=0):
    df = df.sort_values('date', ascending=True)
    dfs = [a[1].reset_index(drop=True) for a in list(df.groupby('FIPS'))]
    for df1 in dfs:
        for col in cols:
            df1[col] = make_monotonic(np.array(df1[col]), epsilon)
    df = pd.concat(dfs).reset_index(drop=True)
    return df
    
def smooth_data(df, cols, window_size=9, order=1, drop_ends = False):
    df = df.copy()
    if window_size%2==0: window_size+=1
    if window_size==1: return df
    FIPS_codes = df['FIPS'].unique()
    for col in cols:
        for fips in FIPS_codes:
            data = np.array(df[df['FIPS']==fips][col])
            smoothed_data = savgol_filter(data, window_size, order)
            df.loc[df['FIPS']==fips, col] = smoothed_data
    if drop_ends: #cut off ends of time series to remove filter distortion
        dates = df['date'].sort_values().unique()[window_size:-window_size] 
        df = df[df['date'].isin(dates)]
    return df

def interpolate_column(df, col, order=1):
    county_dfs = list()
    for county in df['FIPS'].unique():
        a = df[df['FIPS']==county].copy()
        y_full = a[col].values
        x_full = np.array(list(range(len(y_full))))
        b = pd.DataFrame(list(zip(x_full,y_full)), columns=['x','y'])
        b = b.dropna()
        x = b['x'].values
        y = b['y'].values
        f = interp1d(x, y, kind=order)
        a[col] = f(x_full)
        county_dfs.append(a)
    df = pd.concat(county_dfs)
    return df



#### spatial regression stuff

def get_geo_neighbor_edgelist(df, spatial_regrssion_power=-1, max_neighbor_distance=200, #distance in miles
                          dist_data_path='county_distances_2010.parquet', normalize=True):
    counties = df['FIPS'].unique()
    edgelist = pd.read_parquet(dist_data_path) #load data
    edgelist = edgelist[edgelist['mi_to_county']<max_neighbor_distance] #filter on distance
    edgelist = edgelist[edgelist['county1'].isin(counties)] #drop unneeded counties
    edgelist = edgelist[edgelist['county2'].isin(counties)] #drop unneeded counties
    edgelist = edgelist.rename(columns = {'mi_to_county':'weight'})[['county1','county2','weight']] #rename/reorder cols
    edgelist['weight'] = edgelist['weight']**spatial_regrssion_power #take power of distance for spatial weighting
    if normalize:
        norm = edgelist.groupby(by=['county1']).agg({'weight':'sum'}).reset_index().rename(columns = {'weight':'norm'}) #get normalizaiton factor
        edgelist = edgelist.merge(norm, on=['county1'],how='left') 
        edgelist['weight'] = edgelist['weight']/edgelist['norm'] #normalize each county's neighbor weights
    edgelist = edgelist.sort_values('county1')[['county1','county2','weight']].reset_index(drop=True) #make pretty
    return edgelist

def get_retweet_neighbor_edgelist(df, raw_edgelist_path = 'raw_retweet_neighbor_edgelist.parquet',
                                  normalize=True, time_bin_length=8):
    counties = df['FIPS'].unique()
    edgelist = pd.read_parquet(raw_edgelist_path)
    edgelist = edgelist.rename(columns = {'retweeting_FIPS':'county1', 'retweeted_FIPS':'county2'})
    edgelist = edgelist[edgelist['county1'].isin(counties)] #drop unneeded counties
    edgelist = edgelist[edgelist['county2'].isin(counties)] #drop unneeded counties
    edgelist['weight'] = edgelist['n_retweets_per_day']
    if normalize:
        norm = edgelist.groupby(by=['county1']).agg({'weight':'sum'}).reset_index().rename(columns = {'weight':'norm'}) #get normalizaiton factor
        edgelist = edgelist.merge(norm, on=['county1'],how='left') 
        edgelist['weight'] = edgelist['weight']/edgelist['norm'] #normalize each county's neighbor weights
    edgelist = edgelist.sort_values('county1')[['county1','county2','weight']].reset_index(drop=True) #make pretty
    return edgelist

def get_neighbor_mean(df, cols, neighbor_type = 'geo', pop_weighting=False, normalize_edgelist=True):
    
    if isinstance(neighbor_type,pd.DataFrame): #manually override edgelist for testing.
        edgelist = neighbor_type
        neighbor_type = ''
    elif neighbor_type == 'geo':
        edgelist = get_geo_neighbor_edgelist(df, spatial_regrssion_power=-1, max_neighbor_distance=200,
                                             normalize = normalize_edgelist)
    elif neighbor_type == 'retweet':
        edgelist = get_retweet_neighbor_edgelist(df, normalize = normalize_edgelist)
        
    if pop_weighting:
        pops = df[['FIPS','population']].drop_duplicates().reset_index(drop=True)
        edgelist = edgelist.merge(pops.rename(columns={'FIPS':'county1','population':'pop1'}), on='county1', how='inner')
        edgelist = edgelist.merge(pops.rename(columns={'FIPS':'county2','population':'pop2'}), on='county2', how='inner')
        edgelist['weight'] = edgelist['weight']*(edgelist['pop2']/edgelist['pop1'])
        norm = edgelist.groupby(by=['county1']).agg({'weight':'sum'}).reset_index().rename(columns = {'weight':'norm'}) #get normalizaiton factor
        edgelist = edgelist.merge(norm, on=['county1'],how='left') 
        edgelist['weight'] = edgelist['weight']/edgelist['norm'] #normalize each county's neighbor weights
        edgelist = edgelist.drop(columns=['pop1','pop2','norm'])

    for col in cols:
        means_df_list = list()
        for date in df['date'].unique():
            a = (df[df['date']==date][['FIPS',col]]).rename(columns={'FIPS':'county2'})
            means_df = edgelist.copy().merge(a, on='county2', how='left')
            means_df['weight*col'] = means_df['weight']*means_df[col]
            means_df = means_df.groupby('county1').agg({'weight*col':'sum'}).rename(columns = {'weight*col':(col+'_'+neighbor_type+'_neighbor_mean')})
            means_df['date'] = date
            means_df_list.append(means_df)
        means_df = pd.concat(means_df_list).reset_index().rename(columns = {'county1':'FIPS'})
        df = df.merge(means_df, on=['FIPS','date'], how='left')   
    return df

# def get_exposure(df, cols, neighbor_type):  #pop weighted
#     df=df.copy()
#     df1 = df[['FIPS','date','population']+cols].copy()
#     for col in cols:
#         df1[col+'xpop'] = df[col]*df['population']
#         df1 = get_neighbor_mean(df1,[col+'xpop'], 
#                                 neighbor_type=neighbor_type, 
#                                 pop_weighting=(True if neighbor_type=='geo' else False),
#                                 normalize_edgelist=True)
#         df[col+'_exposure'] = df1.eval(col+'xpop_'+neighbor_type+'_neighbor_mean/population')
#     return df

# def get_exposure(df, cols, neighbor_type='retweet'): #unweighted
#     df=df.copy()
#     df1 = df[['FIPS','date','population']+cols].copy()
#     for col in cols:
#         df1 = get_neighbor_mean(df1,[col], 
#                                 neighbor_type=neighbor_type, 
#                                 pop_weighting=(True if neighbor_type=='geo' else False),
#                                 normalize_edgelist=True)
#         df[col+'_exposure'] = df1.eval(col+'_'+neighbor_type+'_neighbor_mean/population')
#     return df


# clustering stuff

def make_county_antivax_clusters(df, cluster_vars, n_clusters, n_PCA_dims = None):
    if 'FIPS_cluster' in df.columns: df = df.drop(columns=['FIPS_cluster'])
    a = df[['FIPS']+cluster_vars].groupby('FIPS').agg('last').reset_index()
    X = a[cluster_vars].values
    X = StandardScaler().fit_transform(X)
    if n_PCA_dims is not None: X = PCA(n_components = n_PCA_dims).fit_transform(X)
    if n_clusters is None: n_clusters = X.shape[0]
    c = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(X)
    a['FIPS_cluster'] = c
    df = df.merge(a[['FIPS','FIPS_cluster']], on='FIPS',how='left')
    return df

def get_cluster_profiles(df, cluster_vars):
    df = df.copy()
    df = df[df['t']==df['t'].max()]
    df = df[['FIPS_cluster']+cluster_vars]
    profiles = df.groupby('FIPS_cluster').agg('mean')
    return profiles



# randomization stuff

def shuffle_col_by_county(data, col, fips_shuffled=None):
    data = data.sort_values(['FIPS','date']).reset_index(drop=True)
    
    fips = pd.Series(data['FIPS'].unique())
    if fips_shuffled is None:
        fips_shuffled = fips.sample(frac=1, replace=False)
    
    fips_remap = dict(zip(fips, fips_shuffled))
    data_shuffled = data[['FIPS','date',col]].copy()
    data_shuffled['FIPS'] = data_shuffled['FIPS'].map(fips_remap)
    
    df = data.drop(columns=col).merge(data_shuffled, on=['FIPS','date'], how='left')
    return df, fips_shuffled

def sample_by_county(data, n=None, frac=None, replace=False):
    fips_sample = pd.Series(data['FIPS'].unique()).sample(n=n, frac=frac, replace=replace)
    dfs = list()
    for county in fips_sample:
        dfs.append(data[data['FIPS']==county])
    df = pd.concat(dfs)
    return df

