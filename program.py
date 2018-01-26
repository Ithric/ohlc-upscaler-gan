import numpy as np
import pandas as pd
import analysis 
import os
import dill as pickle
from common import make_keras_picklable, unzip
from UpscalerModel import UpscalerModel
from datetime import datetime, timedelta
#import toolz as tz
import toolz.curried as tz

NUM_EPOCHS = 10
BATCH_SIZE = 16

# Gotta have some output
if not os.path.exists("./output"):
    os.makedirs("./output")
else:
    from pathlib import Path
    for p in Path("./output").glob("e*.png"):
        p.unlink()


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
        target_rows = df_target.ix[middle[0]+timedelta(days=7):middle[0]+timedelta(days=2*7,microseconds=-1)][col_order].values        
        return [left,middle,right], target_rows

    def filter_valid(sample):
        _,target_rows = sample
        return len(target_rows) == 5

    x, y = tz.pipe(
        tz.sliding_window(3, df_source[col_order].values),
        tz.map(to_sample),
        tz.filter(filter_valid),
        unzip
    )

    return x,y

# analysis.plot_ohlc(df_1w)

# Get the same data in source and target resolution
true_x, true_y = get_samples() 
true_x = [np.array(x) for x in zip(*true_x)]
true_y = [np.array(true_y)]

upscaler_model = UpscalerModel.create_model(5)


for epoch in range(NUM_EPOCHS):
    print("Starting epoch: {}".format(epoch))
    TOTAL_NUM_BATCHES = int((len(true_x[0])+(BATCH_SIZE-1)) / BATCH_SIZE)
    for batch_idx in range(TOTAL_NUM_BATCHES):
        # Pick out the batch
        x = [tx[batch_idx*BATCH_SIZE:(batch_idx+1)*BATCH_SIZE][:,1:] for tx in true_x] 
        y = [ty[batch_idx*BATCH_SIZE:(batch_idx+1)*BATCH_SIZE][:,:,1:] for ty in true_y]

        # Generate "fake" data
        priors = np.random.uniform(size=len(x))
        generated_y = upscaler_model.generate_output(x, priors)
        assert len(generated_y) == 3, "Expected exactly 3 output vector. Got {}".format(len(generated_y))
       
        # Train the critic
        real_samples = list(zip(*[[x[0][i],y[0][i],x[2][i]] for i in range(len(x[0]))]))
        real_samples = [np.array(k) for k in real_samples]
       
        fake_samples = generated_y
        critic_eval_result = upscaler_model.train_critic( real_samples, fake_samples )
        
        # Train the generator (adverserial)
        generator_eval_result = upscaler_model.train_generator(x, priors)
        
        # Print the current results
        print("Epoch: {}, BatchIdx={}/{} results:".format(epoch,batch_idx+1,TOTAL_NUM_BATCHES))
        print("\t Critic: {}".format(critic_eval_result))
        print("\t Generator: {}".format(generator_eval_result))

    # Save the last generated sample
    ohlc_dates = true_y[0][batch_idx*BATCH_SIZE:(batch_idx+1)*BATCH_SIZE]
    ohlc_dates = ohlc_dates[0].reshape((-1,6))[:,:1]    

    ohlc = generated_y[1][-1]
    ohlc = np.concatenate([ohlc_dates,ohlc],axis=1)
    last_ohlc_df = pd.DataFrame(ohlc, columns=["date","open","high","low","close","volume"])
    analysis.plot_ohlc_tofile(last_ohlc_df, "./output/e{}.png".format(epoch))



print("Training complete")

            

            