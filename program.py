import numpy as np
import pandas as pd
import analysis 
import os
from common import unzip
from UpscalerModel import UpscalerModel
from datetime import datetime, timedelta
from sklearn.preprocessing import RobustScaler
from sklearn.utils import shuffle
#import toolz as tz
import toolz.curried as tz
import argparse


DEFAULT_EPOCHS = 250
BATCH_SIZE = 128

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
    df = pd.read_csv("data/gspc.csv", parse_dates=["Date"], index_col=["Date"])

    # Reindex the entire dataset on a "day" period without holes
    df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq='D'))
    df["Date"] = df.index
    df = df.rename(columns={"Date":"date", "Open" : "open", "High" : "high", "Low" : "low", "Close" : "close", "Adj Close" : "adjclose", "Volume" : "volume"})

    # Align the data to start on a Monday to make it easier to work with, and backfill "nan's"
    df = df.ix[df[df.index.dayofweek == 0].iloc[0]["date"]:,:]
    df = df[df.index.dayofweek < 5]
    df = df.fillna(method="backfill", limit=1)
   
    # Create source and target resolutions of the same data
    df_source = analysis.resample_ohlc(df, "1w")
    df_target = df
    
    def to_sample(source_rows):
        source_rows = np.array(source_rows)
        v200davg = np.average(source_rows[:,3].astype(float))
        left, middle, right = source_rows[-3:]
        target_rows = df_target.ix[middle[0]+timedelta(days=7):middle[0]+timedelta(days=2*7,microseconds=-1)][col_order].values   

        # Transform
        left[1:] = left[1:] / v200davg
        middle[1:] = middle[1:] / v200davg
        right[1:] = right[1:] / v200davg
        target_rows = np.concatenate([target_rows[:,:1], target_rows[:,1:]/v200davg], axis=1)

        return v200davg, [left,middle,right], target_rows

    def filter_valid(sample):
        _,_,target_rows = sample
        return len(target_rows) == 5 and not pd.isnull(target_rows).any()

    trend, x, y = tz.pipe(
        tz.sliding_window(28, df_source[col_order].values),
        tz.map(to_sample),
        tz.filter(filter_valid),
        unzip
    )

    return trend, x,y


def run(mode, modelname, forcenew, epochs):
    allow_train = True if mode == "train" or mode == "both" else False
    allow_generate = True if mode == "eval" or mode == "both" else False

    # Get the same data in source and target resolution
    trendline, true_x, true_y = get_samples()     
    true_x = [np.array(x) for x in zip(*true_x)]
    true_y = [np.array(true_y)]

    x_scalers = [RobustScaler() for x in true_x]
    y_scalers = [RobustScaler() for y in true_y]
    [scaler.fit(data[:,1:]) for data,scaler in zip(true_x,x_scalers)]
    [scaler.fit(data[:,:,1:].reshape((-1,5))) for data,scaler in zip(true_y,y_scalers)]

    # Shuffle data
    randstate = np.random.random_integers(0,10000)
    trendline = shuffle(trendline, random_state=randstate)
    true_x = shuffle(*true_x, random_state=randstate)
    true_y = [shuffle(true_y[0], random_state=randstate)]

    if forcenew or not UpscalerModel.exists(modelname):
        upscaler_model = UpscalerModel.create_model(5)
    else:
        upscaler_model = UpscalerModel.load_model(modelname)

    if allow_train:
        for epoch in range(epochs):
            print("Starting epoch: {}".format(epoch))
            critic_generator_advantage = 1.0
            real_fake_advantage = 1.0
            TOTAL_NUM_BATCHES = int((len(true_x[0])+(BATCH_SIZE-1)) / BATCH_SIZE)
            for batch_idx in range(TOTAL_NUM_BATCHES):
                # Pick out the batch
                x = [tx[batch_idx*BATCH_SIZE:(batch_idx+1)*BATCH_SIZE][:,1:] for tx in true_x] 
                y = [ty[batch_idx*BATCH_SIZE:(batch_idx+1)*BATCH_SIZE][:,:,1:] for ty in true_y]

                x = [scaler.transform(tx) for scaler,tx in zip(x_scalers,x)]
                y = [scaler.transform(ty.reshape((-1,5))).reshape(ty.shape) for scaler,ty in zip(y_scalers,y)]

                # Generate "fake" data
                noise_mod = (1.0 - epoch/epochs)
                noised = lambda a: np.random.normal(scale=noise_mod, size=a.shape)+a
                x = [noised(k) for k in x]
                generated_y = upscaler_model.generate_output(x)
                assert len(generated_y) == 3, "Expected exactly 3 output vector. Got {}".format(len(generated_y))
            
                # Train the critic
                real_samples = list(zip(*[[x[0][i],y[0][i],x[2][i]] for i in range(len(x[0]))]))
                real_samples = [np.array(k) for k in real_samples]
            
                fake_samples = generated_y
                critic_eval_result = upscaler_model.train_critic( real_samples, fake_samples, 1.0 / critic_generator_advantage, real_fake_advantage )
                
                # Train the generator (adverserial)
                generator_eval_result = upscaler_model.train_generator(x, critic_generator_advantage)
                
                # Print the current results
                print("Epoch: {}, BatchIdx={}/{} results:".format(epoch,batch_idx+1,TOTAL_NUM_BATCHES))
                print("\t Critic: {}".format(critic_eval_result))
                print("\t Generator: {}".format(generator_eval_result))

                critic_generator_advantage = generator_eval_result[0] / critic_eval_result[1][0]
                real_fake_advantage = critic_eval_result[1][1] / critic_eval_result[0][1]


            if epoch % 10 == 0:
                # Save the last generated sample
                ohlc_dates = true_y[0][batch_idx*BATCH_SIZE:(batch_idx+1)*BATCH_SIZE]
                ohlc_dates = ohlc_dates[0].reshape((-1,6))[:,:1]    
                trend_factor = trendline[batch_idx*BATCH_SIZE]

                ohlc = generated_y[1][-1]
                ohlc = y_scalers[0].inverse_transform(ohlc.reshape((-1,5))).reshape(ohlc.shape)
                ohlc = ohlc * trend_factor
                ohlc = np.concatenate([ohlc_dates,ohlc],axis=1)
                last_ohlc_df = pd.DataFrame(ohlc, columns=["date","open","high","low","close","volume"])
                analysis.plot_ohlc_tofile(last_ohlc_df, "./output/e{}.png".format(epoch))

        if(modelname != "tmp"):
            print("Model saved as: {}".format(upscaler_model.save_model(modelname)))

    if allow_generate:
        # Generate a complete upscaled OHLC series over the entire dataset (true_x)
        x = [scaler.transform(tx[:,1:]) for scaler,tx in zip(x_scalers,true_x)]
        ohlc = upscaler_model.generate_output(x)[1].reshape((-1,5))

        # Get the dates which correspond to the generated output
        ohlc_dates = true_y[0].reshape((-1,6))[:,:1]   
    
        # Transform the upscaled output by inverse scaling and re-applying the trend
        ohlc = y_scalers[0].inverse_transform(ohlc)
        trendline = np.repeat(np.array(trendline),5).reshape((-1,1))
        ohlc = ohlc * trendline
        ohlc = np.concatenate([ohlc_dates,ohlc],axis=1)

        # Build a dataframe from the ohlc data and resample to 1w resolution for comparison with the original
        last_ohlc_df = pd.DataFrame(ohlc, columns=["date","open","high","low","close","volume"])
        last_ohlc_df = last_ohlc_df.set_index(last_ohlc_df["date"])
        last_ohlc_df.sort_index(inplace=True) 
        last_ohlc_df = last_ohlc_df.ix[datetime(2017,1,1):datetime(2018,1,1)]
        last_ohlc_df = analysis.resample_ohlc(last_ohlc_df, "1w").dropna(how='any')
        analysis.plot_ohlc_tofile(last_ohlc_df, "./output/{}_1w.png".format(modelname))
        analysis.plot_ohlc(last_ohlc_df)

        # Plot the original data in a week resolution
        df = pd.read_csv("data/gspc.csv", parse_dates=["Date"])
        df = df.rename(columns={"Date":"date", "Open" : "open", "High" : "high", "Low" : "low", "Close" : "close", "Adj Close" : "adjclose", "Volume" : "volume"})
        df = df.set_index(df["date"])
        df = df.ix[datetime(2017,1,1):datetime(2018,1,1)]
        df = analysis.resample_ohlc(df, "1w")
        analysis.plot_ohlc_tofile(df, "./output/{}_1w_original.png".format(modelname))
        analysis.plot_ohlc(df)



        
    print("Program complete")

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', default="both", action="store", help="Execution mode: 'train', 'eval', 'both'")
    parser.add_argument('-n', '--name', default="tmp", action="store", help="model name")
    parser.add_argument('-f', '--forcenew', action='store_true', help="Discard model if already exists")
    parser.add_argument('-e', '--epochs', default=DEFAULT_EPOCHS, type=int, action='store', help="Number of epochs")
    args = parser.parse_args()
    assert args.mode in ["train", "eval","both"], "invalid mode"

    run(mode=args.mode, modelname=args.name, forcenew=args.forcenew, epochs=args.epochs)

            