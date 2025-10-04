#!/usr/bin/env python
# coding: utf-8

# # Installing and Importing Requisite Packages
# 

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
# import standard libs
from IPython.display import display
from IPython.core.debugger import set_trace as bp
from pathlib import PurePath, Path
import sys
import time
import datetime as dt
from datetime import timedelta
import multiprocessing as mp
from datetime import datetime
from collections import OrderedDict as od
import re
import os
import json
os.environ['THEANO_FLAGS'] = 'device=cpu,floatX=float32'

# import python scientific stack
import pandas as pd
# import pandas_datareader.data as web
from pandas import Timestamp
pd.set_option('display.max_rows', 100)
# from dask import dataframe as dd
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()
import numpy as np
# import scipy.stats as stats
# import statsmodels.api as sm
# from numba import jit
import math
# import pymc3 as pm
# from theano import shared, theano as tt
from sklearn.utils import resample
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc

from itertools import cycle
# from scipy import interp

# import visual tools
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# plt.style.use('seaborn-talk')
plt.style.use('bmh')

#plt.rcParams['font.family'] = 'DejaVu Sans Mono'
#plt.rcParams['font.size'] = 9.5
plt.rcParams['font.weight'] = 'medium'
#plt.rcParams['figure.figsize'] = 10,7
blue, green, red, purple, gold, teal = sns.color_palette('colorblind', 6)

# import util libs
# import pyarrow as pa
# import pyarrow.parquet as pq
from tqdm import tqdm, tqdm_notebook
import warnings
warnings.filterwarnings("ignore")
# import missingno as msno
# from google.colab import drive


# # Defining Necessary Functions

# In[2]:


class MultiProcessingFunctions:
	""" This static functions in this class enable multi-processing"""
	def __init__(self):
		pass

	@staticmethod
	def lin_parts(num_atoms, num_threads):
		""" This function partitions a list of atoms in subsets (molecules) of equal size.
		An atom is a set of indivisible set of tasks.
		"""

		# partition of atoms with a single loop
		parts = np.linspace(0, num_atoms, min(num_threads, num_atoms) + 1)
		parts = np.ceil(parts).astype(int)
		return parts

	@staticmethod
	def nested_parts(num_atoms, num_threads, upper_triangle=False):
		""" This function enables parallelization of nested loops.
		"""
		# partition of atoms with an inner loop
		parts = []
		num_threads_ = min(num_threads, num_atoms)

		for num in range(num_threads_):
			part = 1 + 4 * (parts[-1] ** 2 + parts[-1] + num_atoms * (num_atoms + 1.) / num_threads_)
			part = (-1 + part ** .5) / 2.
			parts.append(part)

		parts = np.round(parts).astype(int)

		if upper_triangle:  # the first rows are heaviest
			parts = np.cumsum(np.diff(parts)[::-1])
			parts = np.append(np.array([0]), parts)
		return parts

	@staticmethod
	def mp_pandas_obj(func, pd_obj, num_threads=24, mp_batches=1, lin_mols=True, **kargs):
		"""	
		:param func: (string) function to be parallelized
		:param pd_obj: (vector) Element 0, is name of argument used to pass the molecule;
						Element 1, is the list of atoms to be grouped into a molecule
		:param num_threads: (int) number of threads
		:param mp_batches: (int) number of batches
		:param lin_mols: (bool) Tells if the method should use linear or nested partitioning
		:param kargs: (var args)
		:return: (data frame) of results
		"""
        # print('mp pandas start')


		if lin_mols:
            # print('lin_parts start')
			parts = MultiProcessingFunctions.lin_parts(len(pd_obj[1]), num_threads * mp_batches)
            # print('lin_parts stop')
		else:
            # print('nested start')
			parts = MultiProcessingFunctions.nested_parts(len(pd_obj[1]), num_threads * mp_batches)
            # print('nested stop')

		jobs = []
		for i in range(1, len(parts)):
			job = {pd_obj[0]: pd_obj[1][parts[i - 1]:parts[i]], 'func': func}
			job.update(kargs)
			jobs.append(job)

		if num_threads == 1:
			out = MultiProcessingFunctions.process_jobs_(jobs)
		else:
			out = MultiProcessingFunctions.process_jobs(jobs, num_threads=num_threads)

		if isinstance(out[0], pd.DataFrame):
			df0 = pd.DataFrame()
		elif isinstance(out[0], pd.Series):
			df0 = pd.Series()
		else:
			return out

		for i in out:
			df0 = df0.append(i)

		df0 = df0.sort_index()
        # print('mp pandas stop')

		return df0

	@staticmethod
	def process_jobs_(jobs):
		""" Run jobs sequentially, for debugging """
		out = []
		for job in jobs:
			out_ = MultiProcessingFunctions.expand_call(job)
			out.append(out_)
		return out

	@staticmethod
	def expand_call(kargs):
		""" Expand the arguments of a callback function, kargs['func'] """
		func = kargs['func']
		del kargs['func']
		out = func(**kargs)
		return out

	@staticmethod
	def report_progress(job_num, num_jobs, time0, task):
		# Report progress as asynch jobs are completed

		msg = [float(job_num) / num_jobs, (time.time() - time0)/60.]
		msg.append(msg[1] * (1/msg[0] - 1))
		time_stamp = str(dt.datetime.fromtimestamp(time.time()))

		msg = time_stamp + ' ' + str(round(msg[0]*100, 2)) + '% '+task+' done after ' + \
			str(round(msg[1], 2)) + ' minutes. Remaining ' + str(round(msg[2], 2)) + ' minutes.'

		if job_num < num_jobs:
			sys.stderr.write(msg+'\r')
		else:
			sys.stderr.write(msg+'\n')

		return

	@staticmethod
	def process_jobs(jobs, task=None, num_threads=24):
		""" Run in parallel. jobs must contain a 'func' callback, for expand_call"""

		if task is None:
			task = jobs[0]['func'].__name__

		pool = mp.Pool(processes=num_threads)
		# outputs, out, time0 = pool.imap_unordered(MultiProcessingFunctions.expand_call,jobs),[],time.time()
		outputs = pool.imap_unordered(MultiProcessingFunctions.expand_call, jobs)
		out = []
		time0 = time.time()

		# Process asyn output, report progress
		for i, out_ in enumerate(outputs, 1):
			out.append(out_)
			MultiProcessingFunctions.report_progress(i, len(jobs), time0, task)

		pool.close()
		pool.join()  # this is needed to prevent memory leaks
		return out


# In[3]:


def get_daily_vol(close, lookback=100):
    """
    :param close: (data frame) Closing prices
    :param lookback: (int) lookback period to compute volatility
    :return: (series) of daily volatility value
    """
    print('Calculating daily volatility for dynamic thresholds')

    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df0 = df0[df0 > 0]
    df0 = (pd.Series(close.index[df0 - 1], index=close.index[close.shape[0] - df0.shape[0]:]))

    df0 = close.loc[df0.index] / close.loc[df0.values].values - 1  # daily returns
    df0 = df0.ewm(span=lookback).std()
    return df0


# In[4]:


def get_t_events(raw_price, threshold):
    """
    :param raw_price: (series) of close prices.
    :param threshold: (float) when the abs(change) is larger than the threshold, the
    function captures it as an event.
    :return: (datetime index vector) vector of datetimes when the events occurred. This is used later to sample.
    """
    print('Applying Symmetric CUSUM filter.')

    t_events = []
    s_pos = 0
    s_neg = 0

    # log returns
    diff = np.log(raw_price).diff().dropna()

    # Get event time stamps for the entire series
    for i in tqdm(diff.index[1:]):
        pos = float(s_pos + diff.loc[i])
        neg = float(s_neg + diff.loc[i])
        s_pos = max(0.0, pos)
        s_neg = min(0.0, neg)

        if s_neg < -threshold:
            s_neg = 0
            t_events.append(i)

        elif s_pos > threshold:
            s_pos = 0
            t_events.append(i)

    event_timestamps = pd.DatetimeIndex(t_events)
    print('get_t_event over.')
    return event_timestamps


# In[5]:


def add_vertical_barrier(t_events, close, num_days=1):
    """
    :param t_events: (series) series of events (symmetric CUSUM filter)
    :param close: (series) close prices
    :param num_days: (int) maximum number of days a trade can be active
    :return: (series) timestamps of vertical barriers
    """
    print( 'vertical_barrier start')
    t1 = close.index.searchsorted(t_events + pd.Timedelta(days=num_days))
    t1 = t1[t1 < close.shape[0]]
    t1 = pd.Series(close.index[t1], index=t_events[:t1.shape[0]])  # NaNs at end
    print('vertical_barrier over')
    return t1


# In[6]:


def apply_pt_sl_on_t1(close, events, pt_sl, molecule):
    """
    :param close: (series) close prices
    :param events: (series) of indices that signify "events" 
    :param pt_sl: (array) element 0, indicates the profit taking level; 
                          element 1 is stop loss level
    :param molecule: (an array) a set of datetime index values for processing
    :return: (dataframe) timestamps at which each barrier was touched
    """
    # apply stop loss/profit taking, if it takes place before t1 (end of event)
    events_ = events.loc[molecule]
    out = events_[['t1']].copy(deep=True)
    if pt_sl[0] > 0:
        pt = pt_sl[0] * events_['trgt']
    else:
        pt = pd.Series(index=events.index)  # NaNs

    if pt_sl[1] > 0:
        sl = -pt_sl[1] * events_['trgt']
    else:
        sl = pd.Series(index=events.index)  # NaNs

    for loc, t1 in events_['t1'].fillna(close.index[-1]).iteritems():
        df0 = close[loc:t1]  # path prices
        df0 = (df0 / close[loc] - 1) * events_.at[loc, 'side']  # path returns
        out.loc[loc, 'sl'] = df0[df0 < sl[loc]].index.min()  # earliest stop loss
        out.loc[loc, 'pt'] = df0[df0 > pt[loc]].index.min()  # earliest profit taking

    return out


# In[7]:


def get_events(close, t_events, pt_sl, target, min_ret, num_threads, 
              vertical_barrier_times=False, side=None):
    """
    :param close: (series) Close prices
    :param t_events: (series) of t_events. 
                     These are timestamps that will seed every triple barrier.
    :param pt_sl: (2 element array) element 0, indicates the profit taking level; 
                  element 1 is stop loss level.
                  A non-negative float that sets the width of the two barriers. 
                  A 0 value means that the respective horizontal barrier will be disabled.
    :param target: (series) of values that are used (in conjunction with pt_sl)
                   to determine the width of the barrier.
    :param min_ret: (float) The minimum target return required for running a triple barrier search.
    :param num_threads: (int) The number of threads concurrently used by the function.
    :param vertical_barrier_times: (series) A pandas series with the timestamps of the vertical barriers.
    :param side: (series) Side of the bet (long/short) as decided by the primary model
    :return: (data frame) of events
            -events.index is event's starttime
            -events['t1'] is event's endtime
            -events['trgt'] is event's target
            -events['side'] (optional) implies the algo's position side
    """
    print('get_events start')
    # 1) Get target
    target = target.loc[target.index.intersection(t_events)]
    target = target[target > min_ret]  # min_ret

    # 2) Get vertical barrier (max holding period)
    if vertical_barrier_times is False:
        vertical_barrier_times = pd.Series(pd.NaT, index=t_events)

    # 3) Form events object, apply stop loss on vertical barrier
    if side is None:
        side_ = pd.Series(1., index=target.index)
        pt_sl_ = [pt_sl[0], pt_sl[0]]
    else:
        side_ = side.loc[target.index]
        pt_sl_ = pt_sl[:2]

    events = pd.concat({'t1': vertical_barrier_times, 'trgt': target, 'side': side_},
                        axis=1)
    events = events.dropna(subset=['trgt'])

    print('MultiProcessing mp_pandas start')
    # Apply Triple Barrier
    df0 = MultiProcessingFunctions.mp_pandas_obj(func=apply_pt_sl_on_t1,
                                                 pd_obj=('molecule', events.index),
                                                 num_threads=num_threads,
                                                 close=close,
                                                 events=events,
                                                 pt_sl=pt_sl_)

    print('MultiProcessing mp_pandas stop')
    events['t1'] = df0.dropna(how='all').min(axis=1)  # pd.min ignores nan

    if side is None:
        events = events.drop('side', axis=1)
    print('get_events stop')
    return events


# In[8]:


def barrier_touched(out_df):
    """
    :param out_df: (DataFrame) containing the returns and target
    :return: (DataFrame) containing returns, target, and labels
    """
    store = []
    for i in np.arange(len(out_df)):
        date_time = out_df.index[i]
        ret = out_df.loc[date_time, 'ret']
        target = out_df.loc[date_time, 'trgt']

        if ret > 0.0 and ret > target:
            # Top barrier reached
            store.append(1)
        elif ret < 0.0 and ret < -target:
            # Bottom barrier reached
            store.append(-1)
        else:
            # Vertical barrier reached
            store.append(0)

    out_df['bin'] = store

    return out_df


# In[9]:


def get_bins(triple_barrier_events, close):
    """
    :param triple_barrier_events: (data frame)
                -events.index is event's starttime
                -events['t1'] is event's endtime
                -events['trgt'] is event's target
                -events['side'] (optional) implies the algo's position side
                Case 1: ('side' not in events): bin in (-1,1) <-label by price action
                Case 2: ('side' in events): bin in (0,1) <-label by pnl (meta-labeling)
    :param close: (series) close prices
    :return: (data frame) of meta-labeled events
    """

    # 1) Align prices with their respective events
    events_ = triple_barrier_events.dropna(subset=['t1'])
    prices = events_.index.union(events_['t1'].values)
    prices = prices.drop_duplicates()
    prices = close.reindex(prices, method='bfill')

    # 2) Create out DataFrame
    out_df = pd.DataFrame(index=events_.index)
    # Need to take the log returns, else your results will be skewed for short positions
    out_df['ret'] = np.log(prices.loc[events_['t1'].values].values) - np.log(prices.loc[events_.index])
    out_df['trgt'] = events_['trgt']

    # Meta labeling: Events that were correct will have pos returns
    if 'side' in events_:
        out_df['ret'] = out_df['ret'] * events_['side']  # meta-labeling

    # Added code: label 0 when vertical barrier reached
    out_df = barrier_touched(out_df)

    # Meta labeling: label incorrect events with a 0
    if 'side' in events_:
        out_df.loc[out_df['ret'] <= 0, 'bin'] = 0

    # Transform the log returns back to normal returns.
    out_df['ret'] = np.exp(out_df['ret']) - 1

    # Add the side to the output. This is useful for when a meta label model must be fit
    tb_cols = triple_barrier_events.columns
    if 'side' in tb_cols:
        out_df['side'] = triple_barrier_events['side']

    out_df

    return out_df


# In[10]:


def bbands(close_prices, window, no_of_stdev):
    # rolling_mean = close_prices.rolling(window=window).mean()
    # rolling_std = close_prices.rolling(window=window).std()
    rolling_mean = close_prices.ewm(span=window).mean()
    rolling_std = close_prices.ewm(span=window).std()

    upper_band = rolling_mean + (rolling_std * no_of_stdev)
    lower_band = rolling_mean - (rolling_std * no_of_stdev)

    return rolling_mean, upper_band, lower_band


# In[11]:


def get_dollar_bars(time_bars, dollar_threshold):

    # initialize an empty list of dollar bars
    dollar_bars = []

    # initialize the running dollar volume at zero
    running_volume = 0

    # initialize the running high and low with placeholder values
    running_high, running_low = 0, math.inf

    # for each time bar...
    for i in range(len(time_bars)):

        # get the timestamp, open, high, low, close, and volume of the next bar
        next_close, next_high, next_low, next_open, next_timestamp, next_volume = [time_bars[i][k] for k in ['close', 'high', 'low', 'open', 'timestamp', 'vol']]

        # get the midpoint price of the next bar (the average of the open and the close)
        midpoint_price = ((next_open) + (next_close))/2

        # get the approximate dollar volume of the bar using the volume and the midpoint price
        dollar_volume = next_volume * midpoint_price

        # update the running high and low
        running_high, running_low = max(running_high, next_high), min(running_low, next_low)

        # if the next bar's dollar volume would take us over the threshold...
        if dollar_volume + running_volume >= dollar_threshold:

            # set the timestamp for the dollar bar as the timestamp at which the bar closed (i.e. one minute after the timestamp of the last minutely bar included in the dollar bar)
            bar_timestamp = next_timestamp + timedelta(minutes=1)

            # add a new dollar bar to the list of dollar bars with the timestamp, running high/low, and next close
            dollar_bars += [{'timestamp': bar_timestamp, 'open': next_open, 'high': running_high, 'low': running_low, 'close': next_close}]

            # reset the running volume to zero
            running_volume = 0

            # reset the running high and low to placeholder values
            running_high, running_low = 0, math.inf

        # otherwise, increment the running volume
        else:
            running_volume += dollar_volume

    # return the list of dollar bars
    return dollar_bars


# In[12]:


## plot for grid search
def plot_search_results(grid):
    """
    Params: 
        grid: A trained GridSearchCV object.
    """
    ## Results from grid search
    results = grid.cv_results_
    means_test = results['mean_test_score']
    stds_test = results['std_test_score']

    ## Getting indexes of values per hyper-parameter
    masks=[]
    masks_names= list(grid.best_params_.keys())
    for p_k, p_v in grid.best_params_.items():
        masks.append(list(results['param_'+p_k].data==p_v))

    params=grid.param_grid

    ## Ploting results
    fig, ax = plt.subplots(1,len(params),sharex='none', sharey='all',figsize=(20,5))
    fig.suptitle('Score per parameter')
    fig.text(0.04, 0.5, 'MEAN SCORE', va='center', rotation='vertical')
    pram_preformace_in_best = {}
    for i, p in enumerate(masks_names):
        m = np.stack(masks[:i] + masks[i+1:])
        pram_preformace_in_best
        best_parms_mask = m.all(axis=0)
        best_index = np.where(best_parms_mask)[0]
        x = np.array(params[p])
        y_1 = np.array(means_test[best_index])
        e_1 = np.array(stds_test[best_index])
        ax[i].errorbar(x, y_1, e_1, linestyle='--', marker='o', label='test')
        ax[i].set_xlabel(p.upper())

    plt.legend()
    plt.show()


# # Implementation

# Import and Format Data

# In[13]:


# mount personal drive
# drive.mount('')


# ## Data Load

# In[14]:


#download the data into your google drive to use the following code
infp=PurePath('BTCUSDT_1min.csv')

#import data and set timestamps as index
data = pd.read_csv(infp)   
data['open_time']= pd.to_datetime((data['open_time']))

data = data[['open_time', 'open', 'high', 'low', 'close', 'volume']]
data.rename(columns = {'open_time':'timestamp', 'open':'open', 'high':'high', 'low':'low','close':'close', 'volume':'vol'},
            inplace = True)

data.tail() #  ~ 2022-09-30 20:25:00


# In[15]:


data.head()


# In[16]:


#download the data into your google drive to use the following code
infp=PurePath('ETHBTC_1min.csv')

#import data and set timestamps as index
data2 = pd.read_csv(infp)   
data2['open_time']= pd.to_datetime((data2['open_time']))

data2 = data2[['open_time', 'open', 'high', 'low', 'close', 'volume']]
data2.rename(columns = {'open_time':'timestamp', 'open':'open', 'high':'high', 'low':'low','close':'close', 'volume':'vol'},
            inplace = True)

data2.head()


# In[17]:


# dollar bar : 5 million threshold
data_dict = data.to_dict('records') # 각 row를 딕셔너리로 해서 리스트 변환
data_dict[0]


# In[18]:


dollar_bars = get_dollar_bars(data_dict, 1000000) #5,000,000 is an arbitrarily selected threshold
data_db = pd.DataFrame(dollar_bars)
data_db = data_db.set_index('timestamp')
data_db.head()


# In[19]:


data = data.set_index('timestamp')
data2 = data2.set_index('timestamp')
data['eth_close'] = data['close'].mul(data2['close']).dropna()
data.head()


# In[20]:


data_db = pd.concat([data_db, data['eth_close']], join = 'inner', axis = 1)
data_db.head()


# ## [Time bar] : Model Apply 

# ### Primary Model

# Create Primary Bollinger Band Model

# In[21]:


import copy
data_w = copy.deepcopy(data)

# compute bands
window = 50
data_w['avg'], data_w['upper'], data_w['lower'] = bbands(data_w['close'], window, no_of_stdev=2)

# compute sides
data_w['side'] = np.nan
long_signals = (data_w['close'] <= data_w['lower'])
short_signals = (data_w['close'] >= data_w['upper'])
data_w.loc[long_signals, 'side'] = 1
data_w.loc[short_signals, 'side'] = -1

print(data_w.side.value_counts())

# Remove Look ahead biase by lagging the signal
data_w['side'] = data_w['side'].shift(1) #다음 1분이 숏이다 롱이다. 편의상 1분이라는 기간을 뒀다. 그 다음 1분동안이 신호다.

# Drop the NaN values from our data set
data_w.dropna(axis=0, how='any', inplace=True)  


# Implement Triple Barriers

# In[ ]:


close = data_w['close']

# determining daily volatility using the last 50 days
daily_vol = get_daily_vol(close=close, lookback=50)

# creating our event triggers using the CUSUM filter 
cusum_events = get_t_events(close, threshold=daily_vol.mean()*0.1)

# adding vertical barriers with a half day expiration window
vertical_barriers = add_vertical_barrier(t_events=cusum_events,
                                         close=close, num_days=0.5)

# determining timestamps of first touch   

pt_sl = [1, 2] # setting profit-take and stop-loss at 1% and 2%          2% 떨어지면 손절 , 1% 이익이 나면 손절 사용자에게 직접 입력받음
min_ret = 0.0005 # setting a minimum return of 0.05%                     0.05% 최소 수익률..


triple_barrier_events = get_events(close=close,
                                  t_events=cusum_events,
                                  pt_sl=pt_sl,
                                  target=daily_vol,
                                  min_ret=min_ret,
                                  num_threads=2,
                                  vertical_barrier_times=vertical_barriers,
                                  side=data_w['side'])


# Add Labels

# In[ ]:


labels = get_bins(triple_barrier_events, data_w['close'])
labels.side.value_counts()


# Evaluating Primary Model

# In[25]:


# creating dataframe of only bin labels
primary_forecast = pd.DataFrame(labels['bin'])

# setting predicted column to 1 
primary_forecast['pred'] = 1
primary_forecast.columns = ['actual', 'pred']

# Performance Metrics
actual = primary_forecast['actual']
pred = primary_forecast['pred']
print(classification_report(y_true=actual, y_pred=pred))

print("Confusion Matrix")
print(confusion_matrix(actual, pred))

print('')
print("Accuracy")
print(accuracy_score(actual, pred))


# #### Graph

# In[164]:


import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots

n = 2; m = 2
label_1 = labels[labels['bin'] == 1]
list_1 = random.sample(list(label_1.index), k = n*m)

fig = make_subplots(
    rows = n, cols = m
)

for idx, st in enumerate(list_1):
  et = triple_barrier_events.loc[st]['t1']
  gr_dt = data.loc[st:et]
  avg, upper, lower = bbands(gr_dt['close'], 20, no_of_stdev=2)
  fig.add_trace(go.Candlestick(x=gr_dt.index,
                open=gr_dt.open, high=gr_dt.high,
                low=gr_dt.low, close=gr_dt.close),
                row = idx//n + 1, col = idx%m + 1)
  fig.add_trace(go.Scatter(x=gr_dt.index, y = avg,
                           line = dict(color='black', width=1)),
                row = idx//n + 1, col = idx%m + 1)
  fig.add_trace(go.Scatter(x=gr_dt.index, y = upper,
                           line = dict(color='blue', width=1)),
                row = idx//n + 1, col = idx%m + 1)
  fig.add_trace(go.Scatter(x=gr_dt.index, y = lower,
                           line = dict(color='red', width=1)),
                row = idx//n + 1, col = idx%m + 1)

  fig.update_xaxes(rangeslider_visible=False, visible = False)

fig.update_layout(title = 'Price change in tripple barrier window ')
fig.update_traces(showlegend = False)
fig.show()



# In[136]:


import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots

n = 2; m = 2
label_1 = labels[labels['bin'] == 0]
list_1 = random.sample(list(label_1.index), k = n*m)

fig = make_subplots(
    rows = n, cols = m
)

for idx, st in enumerate(list_1):
  et = triple_barrier_events.loc[st]['t1']
  gr_dt = data.loc[st:et]
  avg, upper, lower = bbands(gr_dt['close'], 20, no_of_stdev=2)
  fig.add_trace(go.Candlestick(x=gr_dt.index,
                open=gr_dt.open, high=gr_dt.high,
                low=gr_dt.low, close=gr_dt.close),
                row = idx//n + 1, col = idx%m + 1)
  fig.add_trace(go.Scatter(x=gr_dt.index, y = avg,
                           line = dict(color='black', width=1)),
                row = idx//n + 1, col = idx%m + 1)
  fig.add_trace(go.Scatter(x=gr_dt.index, y = upper,
                           line = dict(color='blue', width=1)),
                row = idx//n + 1, col = idx%m + 1)
  fig.add_trace(go.Scatter(x=gr_dt.index, y = lower,
                           line = dict(color='red', width=1)),
                row = idx//n + 1, col = idx%m + 1)

  fig.update_xaxes(rangeslider_visible=False, visible = False)

fig.update_layout(title = 'Price change in tripple barrier window ')
fig.update_traces(showlegend = False)
fig.show()


# ### Secondary Model

# In[26]:


# Get features at event dates
X = data_w.loc[labels.index]
X['rvi'] = (X['open'] - X['close']) / (X['high'] - X['low'])
X = X.loc[:,['close', 'eth_close', 'side', 'rvi']]
y = labels['bin']

# Split data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
X_train = X[X.index.year < 2022]
X_test = X[X.index.year == 2022]
y_train = y[y.index.year < 2022]
y_test = y[y.index.year == 2022]


# In[27]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

# Minmax Scaler
ms = MinMaxScaler() # max = 1 , min = 0
X_train_m = pd.DataFrame(ms.fit_transform(X_train), columns = X_train.columns)
X_test_m = pd.DataFrame(ms.transform(X_test), columns = X_test.columns)

# Standard Scaler
ss = StandardScaler() # mean = 0, var = 1
X_train_s = pd.DataFrame(ss.fit_transform(X_train), columns = X_train.columns)
X_test_s = pd.DataFrame(ss.transform(X_test), columns = X_test.columns)

# Robust Scaler
rs = RobustScaler() # Scaling
X_train_r = pd.DataFrame(rs.fit_transform(X_train), columns = X_train.columns)
X_test_r = pd.DataFrame(rs.transform(X_test), columns = X_test.columns)


# #### Decision Tree (Creating Secondary Model)

# Grid Search

# In[ ]:


from sklearn.model_selection import GridSearchCV

# Setting random forest parameters
RANDOM_STATE = 0

# Fitting our model

grid_params = {
    'criterion': ['entropy', 'gini'], 
    'max_depth': list(range(1,50))
}

dt = DecisionTreeClassifier()
gs = GridSearchCV(dt, grid_params, cv=5, scoring='roc_auc')
gs.fit(X_train_m, y_train)


# In[ ]:


print("Best Parameters : ", gs.best_params_)
print("Best Score : ", gs.best_score_)
print("Best Test Score : ", gs.score(X_test_m, y_test))

print(pd.DataFrame(gs.cv_results_))


# In[ ]:


plot_search_results(gs)


# In[ ]:


from sklearn import tree 
###########
# Setting random forest parameters

clf = tree.DecisionTreeClassifier(criterion = "gini", max_depth = 1)  

# Fitting our model
clf.fit(X_train_m, y_train)

# Performance Metrics
y_pred = clf.predict(X_test_m)
print(classification_report(y_test, y_pred))

print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))

print('')
print("Accuracy")
print(accuracy_score(y_test, y_pred))


# #### Random Forest (Creating Secondary Model)

# In[ ]:


###########
# Setting random forest parameters
n_estimator = 100
depth = 20
RANDOM_STATE = 0

clf = RandomForestClassifier(max_depth=depth, n_estimators=n_estimator,
                            criterion='entropy', class_weight='balanced_subsample',
                            random_state=RANDOM_STATE)

# Fitting our model
clf.fit(X_train_m, y_train)


# Performance Metrics
y_pred = clf.predict(X_test_m)
print(classification_report(y_test, y_pred))

print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))

print('')
print("Accuracy")
print(accuracy_score(y_test, y_pred))


# Grid Search for Random Forest

# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
# Setting random forest parameters
RANDOM_STATE = 0

# Fitting our model

grid_params = {
    'bootstrap': [True, False],
    'max_depth': [5, 10, 15, 20, 25, 30],
    'max_features': ['auto', 'sqrt'],
    'n_estimators': [5, 10, 15, 30, 50, 100]
}

rf = RandomForestClassifier()
gs = GridSearchCV(rf, grid_params, cv=5, scoring='roc_auc')
gs.fit(X_train_s, y_train)


# In[ ]:


print("Best Parameters : ", gs.best_params_)
print("Best Score : ", gs.best_score_)
print("Best Test Score : ", gs.score(X_test, y_test))

print(pd.DataFrame(gs.cv_results_))


# In[ ]:


plot_search_results(gs)


# In[ ]:


###########
# Setting random forest parameters
n_estimator = 5
depth = 5
RANDOM_STATE = 0

clf = RandomForestClassifier(max_depth=depth, n_estimators=n_estimator,
                             bootstrap = False, criterion='entropy', 
                             class_weight='balanced_subsample', max_features = 'sqrt',
                             random_state=RANDOM_STATE)

# Fitting our model
clf.fit(X_train_s, y_train)


# Performance Metrics
y_pred = clf.predict(X_test_s)
print(classification_report(y_test, y_pred))

print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))

print('')
print("Accuracy")
print(accuracy_score(y_test, y_pred))


# #### KNN (Creating Secondary Model)

# Grid Search for KNN

# In[ ]:


# Grid serach for KNN
from sklearn.model_selection import GridSearchCV

grid_params = {
    'n_neighbors' : list(range(3,31)),
    'weights' : ["uniform", "distance"],
    'metric' : ['euclidean', 'manhattan', 'minkowski']
}

knn = KNeighborsClassifier()
gs = GridSearchCV(knn, grid_params, cv=5, scoring='roc_auc')
gs.fit(X_train_m, y_train)


# In[ ]:


print("Best Parameters : ", gs.best_params_)
print("Best Score : ", gs.best_score_)
print("Best Test Score : ", gs.score(X_test_m, y_test))

print(pd.DataFrame(gs.cv_results_))


# In[ ]:


plot_search_results(gs)


# In[28]:


from sklearn.neighbors import KNeighborsClassifier


#  활용
clf = KNeighborsClassifier(n_neighbors = 15, metric = 'manhattan', weights = 'uniform')     

# Fitting our model
clf.fit(X_train_m, y_train)

# Performance Metrics
y_pred = clf.predict(X_test_m)
print(classification_report(y_test, y_pred))

print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))

print('')
print("Accuracy")
print(accuracy_score(y_test, y_pred))


# #### Graph

# In[ ]:


import numpy as np
import copy as cp
import matplotlib.pyplot as plt

import seaborn as sns
from typing import Tuple
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, cross_val_score

def cross_val_predict(model, kfold : KFold, X : np.array, y : np.array) -> Tuple[np.array, np.array, np.array]:

    model_ = cp.deepcopy(model)

    no_classes = len(np.unique(y))

    actual_classes = np.empty([0], dtype=int)
    predicted_classes = np.empty([0], dtype=int)
    predicted_proba = np.empty([0, no_classes]) 

    for train_ndx, test_ndx in kfold.split(X):

        train_X, train_y, test_X, test_y = X[train_ndx], y[train_ndx], X[test_ndx], y[test_ndx]

        actual_classes = np.append(actual_classes, test_y)

        model_.fit(train_X, train_y)
        predicted_classes = np.append(predicted_classes, model_.predict(test_X))

        try:
            predicted_proba = np.append(predicted_proba, model_.predict_proba(test_X), axis=0)
        except:
            predicted_proba = np.append(predicted_proba, np.zeros((len(test_X), no_classes), dtype=float), axis=0)

    return actual_classes, predicted_classes, predicted_proba


# In[ ]:


def plot_confusion_matrix(actual_classes : np.array, predicted_classes : np.array, sorted_labels : list):

    matrix = confusion_matrix(actual_classes, predicted_classes, labels=sorted_labels)

    plt.figure(figsize=(12.8,6))
    sns.heatmap(matrix, annot=True, xticklabels=sorted_labels, yticklabels=sorted_labels, cmap="Blues", fmt="g")
    plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion Matrix')

    plt.show()


# In[ ]:


model = KNeighborsClassifier(n_neighbors = 15, metric = 'manhattan', weights = 'uniform')
kfold = KFold(n_splits=5, random_state=42, shuffle=True)  
actual_classes, predicted_classes, _ = cross_val_predict(model, kfold, X_train_m.to_numpy(), y_train.to_numpy())
plot_confusion_matrix(actual_classes, predicted_classes, [1, 0])


# In[ ]:


cv_results = cross_val_score(model, X, y, cv=kfold, scoring='accuracy', verbose=0)
print(cv_results, cv_results.mean())


# ### Model Save

# In[ ]:


import joblib
joblib.dump(clf, './timebar_knn_model.pkl')


# # Visualization

# ## Buy/Sell Print

# In[33]:


y_pred = pd.Series(y_pred, index = X_test.index, name = 'y_pred')
y_proba = pd.Series(clf.predict_proba(X_test_m)[:,1], index = X_test.index, name = 'y_proba')
df_pred = pd.concat([X_test, y_test, y_pred, y_proba], axis = 1)
df_pred


# In[227]:


simul = data[data.index.year == 2022]
buy_list = df_pred[df_pred['y_pred'] == 1].index
sell_list = triple_barrier_events.loc[buy_list]['t1']
simul['num_coin'] = pd.Series([0]*len(simul), index = simul.index)

simul


# In[228]:


for b_idx, s_idx in zip(buy_list, sell_list):
  prd = simul.loc[b_idx:s_idx][:-1]
  num_coins = df_pred.loc[b_idx]['y_proba']
  prd['num_coin'] += (num_coins * prd['close'])   
  print(f'Buy Bitcoins : {np.round(num_coins, 2)} coins at {str(b_idx)} dollar [{simul.close.loc[b_idx]}]')
  print(f'Sell Bitcoins : {np.round(num_coins, 2)} coins at {str(s_idx)} dollar [{simul.close.loc[s_idx]}]')
print(f'Cryptocurrency Exchanges : Buy [{len(buy_list)}] Sell [{len(sell_list)}]')


# In[ ]:


def buy_or_sell(df):
  buy_df = df[df['y_pred'] == 1]

  for i in range(len(buy_df)):
    row = buy_df.iloc[i]
    next_rows = df.loc[row.name:].iloc[1:]
    for x in range(len(next_rows)-1):
      if next_rows['y_proba'].iloc[x+1] > next_rows['y_proba'].iloc[x]:
        next_row = next_rows.iloc[x+1]
        break




# In[170]:


import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots

n = 2; m = 2
label_1 = df_pred[(df_pred['bin'] == 0) & (df_pred['y_pred'] == 1)]
list_1 = random.sample(list(label_1.index), k = n*m)

fig = make_subplots(
    rows = n, cols = m
)

for idx, st in enumerate(list_1):
  et = triple_barrier_events.loc[st]['t1']
  gr_dt = data.loc[st:et]
  avg, upper, lower = bbands(gr_dt['close'], 20, no_of_stdev=2)
  fig.add_trace(go.Candlestick(x=gr_dt.index,
                open=gr_dt.open, high=gr_dt.high,
                low=gr_dt.low, close=gr_dt.close),
                row = idx//n + 1, col = idx%m + 1)
  fig.add_trace(go.Scatter(x=gr_dt.index, y = avg,
                           line = dict(color='black', width=1)),
                row = idx//n + 1, col = idx%m + 1)
  fig.add_trace(go.Scatter(x=gr_dt.index, y = upper,
                           line = dict(color='blue', width=1)),
                row = idx//n + 1, col = idx%m + 1)
  fig.add_trace(go.Scatter(x=gr_dt.index, y = lower,
                           line = dict(color='red', width=1)),
                row = idx//n + 1, col = idx%m + 1)

  fig.update_xaxes(rangeslider_visible=False, visible = False)

fig.update_layout(title = 'Price change in tripple barrier window ')
fig.update_traces(showlegend = False)
fig.show()


# In[ ]:


buy_or_sell(df_pred)


# ## Cumulative log return graph

# In[31]:


X = data[data.index.year == 2022]
profit = X['close'].pct_change()
profit_acc = (1 + profit).cumprod() - 1
profit


# In[93]:


X = data[data.index.year == 2022]

btc_prf = X['close'].pct_change()
btc_prf_cum = (1 + btc_prf).cumprod() - 1
# log_profit = np.log(profit+1)
# log_profit_acc1 = log_profit.cumsum()

# account return
btc_prf_ac = X['close'].pct_change()
buy_df = df_pred[df_pred['y_pred'] == 1]
idx_srs = vertical_barriers.loc[buy_df.index]
s_idx = idx_srs.index
e_idx = idx_srs


for i in range(len(idx_srs)): 
  if i > 0:
    btc_prf_ac.loc[e_idx[i-1]:s_idx[i]][1:] = 0

  if i < len(idx_srs)-1:
    btc_prf_ac.loc[e_idx[i]:s_idx[i+1]][1:] = 0

btc_prf_ac_cum = (1 + btc_prf_ac).cumprod() - 1


# In[ ]:


X = data2[data2.index.year == 2022]
profit = X['close'].pct_change()
profit_acc = (1 + profit).cumprod() - 1
log_profit = np.log(profit+1)
log_profit_acc2 = log_profit.cumsum()


# In[95]:


fig = plt.figure(figsize=(20,10)) ## 캔버스 생성
fig.set_facecolor('white') ## 캔버스 색상 설정
ax = fig.add_subplot() ## 그림 뼈대(프레임) 생성

ax.spines['right'].set_visible(False) ## 오른쪽 축 숨김
ax.spines['top'].set_visible(False) ## 위쪽 축 숨김

args_plot = [btc_prf_cum.index, btc_prf_cum] ## 데이터 인자
config_plot = dict( ## 키워드 인자
    color= 'orange', # 선 색깔
    linestyle='solid', # 선 스타일
    linewidth=1, # 선 두께 
)

ax.plot(label = 'BTC', *args_plot,**config_plot) ## 선그래프 생성

args_plot2 = [btc_prf_ac_cum.index, btc_prf_ac_cum] ## 데이터 인자
config_plot2 = dict( ## 키워드 인자
    color= 'green', # 선 색깔
    linestyle='solid', # 선 스타일
    linewidth=1, # 선 두께 
)

ax.plot(label = 'BTC_account', *args_plot2,**config_plot2) ## 선그래프 생성

ax.axhline(0, ls = '--', color = 'gray')
# ax.text(0,mean_sales+10,f'Mean of Sales : {mean_sales}',fontsize=13) ## 평균 매출 텍스트 출력

ax.legend(loc='upper right', fontsize=15) ## 범례 생성

plt.xticks(rotation=45) ## x축 눈금 라벨 설정 - 45도 회전 
plt.title('Cumulative log return',fontsize=20) ## 타이틀 설정
plt.show()


# In[ ]:




