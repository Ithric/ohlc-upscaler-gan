#!/usr/bin/env python
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, WeekdayLocator,\
    DayLocator, MONDAY
from matplotlib.finance import quotes_historical_yahoo_ohlc, candlestick_ohlc
from matplotlib.dates import date2num
import toolz.curried as tz
import numpy as np

def ohlc_collapse(ohlc_rows):
    return np.array([ohlc_rows[0][0], np.max(ohlc_rows[:,1]), np.min(ohlc_rows[:,2]), ohlc_rows[-1,3], np.sum(ohlc_rows[:,4])])

def resample_ohlc(dfquotes, period):
    ohlc_dict = {                                                                                                             
        'open':'first',                                                                                                    
        'high':'max',                                                                                                       
        'low':'min',                                                                                                        
        'close': 'last',                                                                                                    
        'volume': 'sum',
        'date' : 'first'
        }

    return dfquotes.resample(period, how=ohlc_dict, closed='left', label='left')

def __plot_ohlc(dfquotes):
    dfquotes = dfquotes.set_index(dfquotes["date"])
    quotes = []
    for t,q in zip(date2num(dfquotes.index.to_pydatetime()), dfquotes[["open","high","low","close"]].values.tolist()):
        quotes.append([t]+q)

    mondays = WeekdayLocator(MONDAY)        # major ticks on the mondays
    alldays = DayLocator()              # minor ticks on the days
    weekFormatter = DateFormatter('%b %d')  # e.g., Jan 12
    dayFormatter = DateFormatter('%d')      # e.g., 12

    # Set up a 5x1 grid, mapped to a visual 2x1 grid
    fig = plt.figure()
    ax1 = plt.subplot2grid( (5,1), (0,0), rowspan=4 )
    ax2 = plt.subplot2grid( (5,1), (4,0), rowspan=1 )
    
    # Render the ohlc plot
    candlestick_ohlc(ax1, quotes)
    ax1.xaxis_date()
    ax1.autoscale_view()
    plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')

    # Render the volume plot
    vols = dfquotes["volume"].values
    dates = dfquotes["date"].values
    ax2.bar(dates,vols)

def plot_ohlc_tofile(dfquotes, filename):
    __plot_ohlc(dfquotes)
    plt.savefig(filename)
    plt.clf()
    plt.close()

def plot_ohlc(dfquotes):
    __plot_ohlc(dfquotes)
    plt.show()

def is_valid_ohlc(ohlc_row):
    """ ohlc format: (open,high,low,close,volume) """
    _open, _high, _low, _close, _volume = ohlc_row
    isin_bunds = lambda v: v >= _low and v <= _high
    return _high >= _low and isin_bunds(_open) and isin_bunds(_close) and _volume >= 0

def group_valid(ohlc):
    groups = tz.groupby(is_valid_ohlc, ohlc)
    valid_ohlc = groups[True] if True in groups else []
    invalid_ohlc = groups[False] if False in groups else []
    return valid_ohlc, invalid_ohlc

def calculate_ohlc_stats(ohlc):
    valid_ohlc, invalid_ohlc = group_valid(ohlc)
    return len(valid_ohlc), len(invalid_ohlc), len(valid_ohlc) / len(ohlc)
    