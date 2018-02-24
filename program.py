import numpy as np
import pandas as pd
import analysis 
import os
from common import unzip
from UpscalerModel import UpscalerModel
from datetime import datetime, timedelta


#import toolz as tz
import toolz.curried as tz
import argparse
import data


DEFAULT_EPOCHS = 250
BATCH_SIZE = 128

# Gotta have some output
if not os.path.exists("./output"):
    os.makedirs("./output")
else:
    from pathlib import Path
    for p in Path("./output").glob("e*.png"):
        p.unlink()


def run(mode, modelname, forcenew, epochs):
    allow_train = True if mode == "train" or mode == "both" else False
    allow_generate = True if mode == "eval" or mode == "both" else False
    
    # Get the samples and 'work it'    
    sample_batches = list(tz.pipe(
        data.get_samples("s&p500", "1d", datetime(1980,1,1), datetime(2018,1,1), random_state=np.random.random_integers(0,234234) ),
        data.samples_to_batch_generator(128)))

    # Load or create the model
    if forcenew or not UpscalerModel.exists(modelname):
        upscaler_model = UpscalerModel.create_model(5)
    else:
        upscaler_model = UpscalerModel.load_model(modelname)

    if allow_train:
        for epoch in range(epochs):
            print("Starting epoch: {}".format(epoch))
            critic_generator_advantage = 1.0
            
            batch_idx = 0
            for y_time, x, y, batch_unscaler in sample_batches:
                # Generate "fake" data
                noise_mod = (1.0 - epoch/epochs) * 0.2
                noised = lambda a: np.random.normal(scale=noise_mod, size=a.shape)+a
                x = [noised(k) for k in x]
                generated_y = upscaler_model.generate_output(x)
                assert len(generated_y) == 3, "Expected exactly 3 output vector. Got {}".format(len(generated_y))
            
                # Train the critic 
                real_samples = y
                fake_samples = generated_y
                critic_eval_result = upscaler_model.train_critic( real_samples, fake_samples, 1.0 / critic_generator_advantage )
                
                # Train the generator (adverserial)
                generator_eval_result = upscaler_model.train_generator(x, critic_generator_advantage)
                
                # Print the current results
                print("Epoch: {}, BatchIdx={} results:".format(epoch,batch_idx+1))
                print("\t Critic: {}".format(critic_eval_result))
                print("\t Generator: {}".format(generator_eval_result))
                ohlc = batch_unscaler(y=generated_y)[0]
                print("\t Valid/Invalid: {} vs {} => {:.2%}%".format(*analysis.calculate_ohlc_stats(ohlc)))

                # Apply another level of training to the critic to deter invalid OHLC
                ohlc_validvec = tz.pipe(ohlc, tz.map(analysis.is_valid_ohlc), list)            
                invalid_ohlc_samples_x = tz.pipe(zip(*generated_y+[ohlc_validvec]),
                    tz.filter(lambda t: t[-1] == False ),
                    tz.map(lambda t: t[:-1]),
                    unzip, tz.map(np.array), list)
                inv_loss = upscaler_model.train_critic_invalid(invalid_ohlc_samples_x)                
                print("\t Invalid loss: {} ({}# samples)".format(inv_loss, len(invalid_ohlc_samples_x[0])))

                critic_generator_advantage = critic_eval_result[1][1] #generator_eval_result[0] / critic_eval_result[1][0]
                batch_idx = batch_idx +1 


            if epoch % 10 == 0:
                # Save the last generated sample
                y_time = y_time.reshape(y[1].shape[:-1]+(1,))
                ohlc = batch_unscaler(y=generated_y)[1]
                ohlc = np.concatenate([y_time, ohlc], axis=2)[-1]                
                last_ohlc_df = pd.DataFrame(ohlc, columns=["date","open","high","low","close","volume"])
                analysis.plot_ohlc_tofile(last_ohlc_df, "./output/e{}.png".format(epoch))

        if(modelname != "tmp"):
            print("Model saved as: {}".format(upscaler_model.save_model(modelname)))

    if allow_generate:
        def best_of_group(ohlc_group):
            """ Select 1 OHLC row per candiate which is valid """
            valid_ohlc_rows = tz.pipe(ohlc_group,
                tz.filter(lambda p: p[0]),
                list)
            if any(valid_ohlc_rows): return valid_ohlc_rows[0][1]
            else: return ohlc_group[0][1]

        NUM_CANDIATES = 25
        vohlc = []
        for y_time, x, y, batch_unscaler in sample_batches:        
            x = [np.repeat(tx, NUM_CANDIATES, axis=0) for tx in x]
            x = [tx+np.random.normal(size=tx.shape, scale=0.05) for tx in x]        
            ohlc = upscaler_model.generate_output(x)
            
            ohlc_candidate_vecs = [batch_unscaler(y=[ox[k::NUM_CANDIATES] for ox in ohlc])[1] for k in range(NUM_CANDIATES)]
            ohlc = ohlc[1]
            for k in range(NUM_CANDIATES):
                ohlc[k::NUM_CANDIATES] = ohlc_candidate_vecs[k]                
            ohlc = [item for sublist in ohlc for item in sublist]
          
             # OHLC contains NUM_CANDIATES per day - rebuild the series by picking the first valid candidate per day
            ohlc = tz.pipe(ohlc,
                tz.map(lambda ohlc_row: (analysis.is_valid_ohlc(ohlc_row), ohlc_row)),
                tz.partition(NUM_CANDIATES),
                tz.map(best_of_group),
                list, np.array
            )
            # Re-apply the time axis
            ohlc = np.concatenate([y_time.reshape(-1,1), ohlc], axis=1)
            vohlc.extend(ohlc)        
       
        # high[2], low[3]
        print("\n\nValid/Invalid: {} / {} => {:.2%}%".format(*analysis.calculate_ohlc_stats(ohlc[:,1:])))

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

            