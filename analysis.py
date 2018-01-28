#!/usr/bin/env python
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, WeekdayLocator,\
    DayLocator, MONDAY
from matplotlib.finance import quotes_historical_yahoo_ohlc, candlestick_ohlc
from matplotlib.dates import date2num

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