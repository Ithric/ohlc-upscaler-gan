import numpy as np
import pandas as pd
import analysis 
import os
import dill as pickle
from common import make_keras_picklable
from UpscalerModel import UpscalerModel
from datetime import datetime, timedelta
import toolz as tz

NUM_EPOCHS = 1
BATCH_SIZE = 16


def get_samples():
    """ Return samples on the format [(source:left,middle,right),..], [target] """
    col_order = ["date","open","high","low","close","volume"]
    df = pd.read_csv("gspc.csv", parse_dates=["Date"])
    df = df.rename(columns={"Date":"date", "Open" : "open", "High" : "high", "Low" : "low", "Close" : "close", "Adj Close" : "adjclose", "Volume" : "volume"})
    df = df.set_index(df["date"])
    df = df.ix[datetime(2017,3,1):]
    df_source = analysis.resample_ohlc(df, "1w")
    df_target = df
    
    def to_sample(source_rows):
        left, middle, right = source_rows
        target_rows = df_target.ix[middle[0]+timedelta(days=7):middle[0]+timedelta(days=2*7,microseconds=-1)].values
        return [left,middle,right], target_rows

    def filter_valid(sample):
        _,target_rows = sample
        return len(target_rows) == 5

    x, y = zip(*filter(filter_valid,map(to_sample, tz.sliding_window(3, df_source[col_order].values))))

    return x,y

# analysis.plot_ohlc(df_1w)

# Get the same data in source and target resolution
true_x, true_y = get_samples() 

upscaler_model = UpscalerModel.create_model()


for epoch in range(NUM_EPOCHS):
    print("Starting epoch: {}".format(epoch))
    TOTAL_NUM_BATCHES = int(len(true_x) / BATCH_SIZE)
    for batch_idx in range(TOTAL_NUM_BATCHES):
        # Pick out the batch
        x = true_x[batch_idx*BATCH_SIZE:(batch_idx+1)*BATCH_SIZE]
        y = true_y[batch_idx*BATCH_SIZE:(batch_idx+1)*BATCH_SIZE]

        # Generate "fake" data
        priors = np.random.uniform(size=len(x))
        generated_y = upscaler_model.generate_output(x, priors)

        # Train the critic
        real_samples = (x,y)
        fake_samples = (x,generated_y)
        critic_eval_result = upscaler_model.train_critic( real_samples, fake_samples )

        # Train the generator (adverserial)
        generator_eval_result = upscaler_model.train_generator(x, priors)

        # Print the current results
        print("Epoch: {}, BatchIdx={}/{} results:")
        print("\t Critic: {}".format(critic_eval_result))
        print("\t Generator: {}".format(generator_eval_result))



            

            