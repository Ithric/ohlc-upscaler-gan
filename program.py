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

NUM_EPOCHS = 250
BATCH_SIZE = 64

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
    #df = df.ix[datetime(2017,3,1):]
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


def run(mode, modelname, forcenew, epochs):
    # Get the same data in source and target resolution
    true_x, true_y = get_samples() 
    true_x = [np.array(x) for x in zip(*true_x)]
    true_y = [np.array(true_y)]

    x_scalers = [RobustScaler() for x in true_x]
    y_scalers = [RobustScaler() for y in true_y]
    [scaler.fit(data[:,1:]) for data,scaler in zip(true_x,x_scalers)]
    [scaler.fit(data[:,:,1:].reshape((-1,5))) for data,scaler in zip(true_y,y_scalers)]

    true_x = shuffle(true_x, random_state=4523)
    true_y = shuffle(true_y, random_state=4523)

    if forcenew or not UpscalerModel.exists(modelname):
        upscaler_model = UpscalerModel.create_model(5)
    else:
        upscaler_model = UpscalerModel.load_model(modelname)


    for epoch in range(epochs):
        print("Starting epoch: {}".format(epoch))
        TOTAL_NUM_BATCHES = int((len(true_x[0])+(BATCH_SIZE-1)) / BATCH_SIZE)
        for batch_idx in range(TOTAL_NUM_BATCHES):
            # Pick out the batch
            x = [tx[batch_idx*BATCH_SIZE:(batch_idx+1)*BATCH_SIZE][:,1:] for tx in true_x] 
            y = [ty[batch_idx*BATCH_SIZE:(batch_idx+1)*BATCH_SIZE][:,:,1:] for ty in true_y]

            x = [scaler.transform(tx) for scaler,tx in zip(x_scalers,x)]
            y = [scaler.transform(ty.reshape((-1,5))).reshape(ty.shape) for scaler,ty in zip(y_scalers,y)]

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

        if epoch % 10 == 0:
            # Save the last generated sample
            ohlc_dates = true_y[0][batch_idx*BATCH_SIZE:(batch_idx+1)*BATCH_SIZE]
            ohlc_dates = ohlc_dates[0].reshape((-1,6))[:,:1]    

            ohlc = generated_y[1][-1]
            ohlc = y_scalers[0].inverse_transform(ohlc.reshape((-1,5))).reshape(ohlc.shape)
            ohlc = np.concatenate([ohlc_dates,ohlc],axis=1)
            last_ohlc_df = pd.DataFrame(ohlc, columns=["date","open","high","low","close","volume"])
            analysis.plot_ohlc_tofile(last_ohlc_df, "./output/e{}.png".format(epoch))

    if(modelname != "tmp"):
        print("Model saved as: {}".format(upscaler_model.save_model(modelname)))
        
    print("Program complete")

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', default="both", action="store", help="Execution mode: 'train', 'generate'")
    parser.add_argument('-n', '--name', default="tmp", action="store", help="model name")
    parser.add_argument('-f', '--forcenew', action='store_true', help="Discard model if already exists")
    parser.add_argument('-e', '--epochs', default=NUM_EPOCHS, type=int, action='store', help="Number of epochs")
    args = parser.parse_args()
    assert args.mode in ["train", "eval","both"], "invalid mode"

    run(mode=args.mode, modelname=args.name, forcenew=args.forcenew, epochs=args.epochs)

            