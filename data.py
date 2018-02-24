import pandas as pd
import numpy as np
from toolz.curried import curry
import toolz.curried as tz
import analysis
from common import *
from datetime import datetime, timedelta
from functools import partial
from sklearn.utils import shuffle
from sklearn.preprocessing import RobustScaler, StandardScaler

def read_ohlc(ticker, ohlc_period):
    """ Return samples on the format [(source:left,middle,right),..], [target] """
    col_order = ["date","open","high","low","close","volume"]
    df = pd.read_csv("data/gspc.csv", parse_dates=["Date"], index_col=["Date"])

    # Reindex the entire dataset on a "day" period without holes
    df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq='D'))
    df["Date"] = df.index
    df = df.rename(columns={"Date":"date", "Open" : "open", "High" : "high", "Low" : "low", "Close" : "close", "Adj Close" : "adjclose", "Volume" : "volume"})
    return df

@curry
def stream_ohlc_by_week(ohlc_df, include_date, sampler):
    """ Group the OHLC data into weeks """
    if not include_date: ohlc_df = ohlc_df[["open", "high", "low", "close", "volume"]]
    else: ohlc_df = ohlc_df[["date", "open", "high", "low", "close", "volume"]]
    ohlc_df = ohlc_df.fillna(method="backfill", limit=2)

    for dt in sampler:
        assert dt.weekday() == 0, "Week sampler required the sample time to be on a monday"
        yield ohlc_df.ix[dt:dt+timedelta(days=6)].values

@curry
def ohlc_sampler(avg_window_size):
    def inner(avg_window_size, df_source_seq):
        """
        Arguments:
        - avg_window_size: Number of periods (of target resolution) to average over
        """        

        def to_sample(ohlc_data):
            if any(map(lambda p: len(p) == 0 or p is None or np.isnan(p).any(), ohlc_data)): return None
         
            v200davg = np.average([np.average(t[:,3],axis=0) for t in ohlc_data], axis=0) # Average the close
            
            # Pick out high rew: (left, middle, right) OHLC, and build the low res ohlc (low_res_ohlc)
            left, middle, right = [k[:5] / v200davg for k in ohlc_data[-3:]] 

            x = list(map(analysis.ohlc_collapse, [left, middle, right]))
            y = [x[0],middle,x[2]]

            return v200davg, x, y

        return tz.pipe(
            tz.sliding_window(avg_window_size, df_source_seq),
            tz.map(to_sample))
    return partial(inner, avg_window_size)


def time_series_sampler(from_dt, to_dt, freq):
    tslist = pd.date_range(from_dt, to_dt, freq=freq)
    yield from tslist

def get_samples(ticker, ohlc_period, from_dt, to_dt, random_state=None):
    """ Read  """
    sample_generator = list(time_series_sampler(from_dt, to_dt, "W-MON"))

    # OHLC stream: Create a "1d" to "1w" stream
    ohlc_stream = tz.pipe(sample_generator,
        stream_ohlc_by_week(read_ohlc(ticker, ohlc_period), False),
        ohlc_sampler(avg_window_size=4))

    # OHLC prediction time stream
    ohlc_timestream = tz.pipe(sample_generator,
        stream_ohlc_by_week(read_ohlc(ticker, ohlc_period), True),
        tz.map(lambda values: values[:5,0]), # pick out the date
        list)

    # Create a common stream and filter it
    def sample_filter(sample):
        if any(map(lambda p: p is None, sample)): return False
        z, (_,x,y) = sample        
        return not any([pd.isnull(tx).any() for tx in x]) and not any([pd.isnull(ty).any() for ty in y])

    z, data_stream = tz.pipe(zip(ohlc_timestream, ohlc_stream),
        tz.filter(sample_filter),
        unzip)
    trend, x, y = list(unzip(data_stream))

    # Reshape some stuff    
    trend = np.array(trend).reshape((-1,1))
    x = [np.array(k) for k in list(unzip(x))]
    y = [np.array(k) for k in list(unzip(y))]
    z = np.array(z)

    def scale_it(vec):
        scaler = RobustScaler()
        vec_scaled = scaler.fit_transform(vec.reshape(-1,5)).reshape(vec.shape)
        return vec_scaled, scaler

    def unscale_it(t):        
        vec,scaler = t
        return scaler.inverse_transform(vec.reshape(-1,5)).reshape(vec.shape)

    @curry
    def apply_trend(trend, vec):  
        if len(trend.shape) != len(vec.shape):            
            return (vec.reshape(vec.shape[0],-1) * trend).reshape(vec.shape)
        else:
            return vec * trend

    x_scaled, xscalers = zip(*map(scale_it, x))
    y_scaled, yscalers = zip(*map(scale_it, y))
   
    def unscaler(xscalers, yscalers, trend, from_idx, to_idx, x=None, y=None):
        assert (x is not None or y is not None), "Both X and Y cannot be None"        
        def restore_it(vecs, scalers, trend):
            restore_pipe = lambda x: tz.pipe(x, unscale_it, apply_trend(trend))
            return list(map(restore_pipe, zip(vecs,scalers)))

        if x is not None and xscalers is not None: x = restore_it(x, xscalers, trend[from_idx:to_idx])
        if y is not None and yscalers is not None: y = restore_it(y, yscalers, trend[from_idx:to_idx])
    
        if y is not None and x is not None: return (x,y)
        else: return x or y
    
    # Shuffle the data
    # if random_state is not None:
    #     trend = shuffle(trend, random_state=random_state)
    #     x = shuffle(*x, random_state=random_state)
    #     y = [shuffle(y[0], random_state=random_state)]
    
    return z, x_scaled, y_scaled, partial(unscaler,xscalers,yscalers,trend)

@curry
def samples_to_batch_generator(batch_size, sample_data):
    z, x, y, unscaler = sample_data

    TOTAL_NUM_BATCHES = int((len(x[0])+(batch_size-1)) / batch_size)
    for batch_idx in range(TOTAL_NUM_BATCHES):
        a,b = (batch_idx*batch_size, (batch_idx+1)*batch_size)
        batch_z = z[a:b]
        batch_x = [tx[a:b] for tx in x]
        batch_y = [ty[a:b] for ty in y]

        yield batch_z, batch_x, batch_y, partial(unscaler, batch_idx*batch_size, (batch_idx+1)*batch_size)


