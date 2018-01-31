import numpy as np
import dill as pickle
from common import make_keras_picklable, unzip
import os
import analysis
import toolz.curried as tz

make_keras_picklable()

BOOSTING_HYSTERESIS_RATIO = 1.2

class UpscalerModel(object):
    def __init__(self, actor, critic):
        self.__generator = actor
        self.__critic = critic
     
        self.__critic_trainer = UpscalerModel.__create_critic_trainer(critic)
        self.__adverserial_trainer = UpscalerModel.__create_adverserial_trainer(actor, critic)

    def __create_critic_trainer(critic):
        from keras.optimizers import RMSprop
        
        optimizer = RMSprop(lr=0.0002, decay=6e-8)
        critic.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return critic

    def __create_adverserial_trainer(actor, critic):
        from keras.layers import Input
        from keras.optimizers import Adam
        from keras.models import Model
        
        ohlc_left_input = Input(shape=(5,))
        ohlc_middle_input = Input(shape=(5,))
        ohlc_right_input = Input(shape=(5,))

        y = actor([ohlc_left_input,ohlc_middle_input,ohlc_right_input])
        y = critic(y)
        y.Trainable = False # Dont wanna train the critic AND actor at the same time

        model = Model(inputs=[ohlc_left_input,ohlc_middle_input,ohlc_right_input], outputs=y)
        optimizer = Adam(lr=0.0001)
        model.compile(loss="mse", optimizer=optimizer, metrics=["accuracy"])
        return model

    def create_model(upscaling_factor):
        from keras.layers import Input, Dense, LSTM, concatenate, Flatten, LeakyReLU, GaussianNoise, Dropout, Bidirectional, Reshape
        from keras.models import Model        

        def create_actor():
            """ map data:  (ohlc_t-1, ohlc_t0, ohlc_t+1) -> (ohlc_t-1, ohlc_t0 @ upscaling_factor, ohlc_t+1) 
                map shape: [(5,),(5,),(5,)]              -> [(5,),(upscaling_factor,5),(5,)]

            Map a triple of OHLC onto a higher estimated resolution in place of ohlc_t0
            """
            ohlc_left_input = Input(shape=(5,))
            ohlc_middle_input = Input(shape=(5,))
            ohlc_right_input = Input(shape=(5,))

            # Upscale the middle
            middle = Dense(4*upscaling_factor*5, activation="linear")(ohlc_middle_input)
            middle = LeakyReLU(alpha=0.3)(middle)
            middle = Dense(3*upscaling_factor*5, activation="linear")(middle)
            middle = LeakyReLU(alpha=0.3)(middle)
            middle = Dense(2*upscaling_factor*5, activation="linear")(middle)
            middle = Dense(upscaling_factor*5, activation="tanh")(middle)
            middle = Reshape(target_shape=(upscaling_factor,5))(middle)

            model = Model(inputs=[ohlc_left_input, ohlc_middle_input, ohlc_right_input], outputs=[ohlc_left_input,middle,ohlc_right_input])
            return model

        def create_critic():
            """ map data:  (ohlc_t-1, ohlc_t0 @ upscaling_factor, ohlc_t+1) -> (real: yes/no)
                map shape: [(5,),(upscaling_factor,5),(5,)]                 -> (1,)
            """
            ohlc_left_input = Input(shape=(5,))
            ohlc_middle_input = Input(shape=(upscaling_factor,5))
            ohlc_right_input = Input(shape=(5,))

            ohlc_middle = Flatten()(ohlc_middle_input)
            y = concatenate([ohlc_left_input, ohlc_middle, ohlc_right_input])
            y = Dense(128, activation="linear")(y)
            y = LeakyReLU()(y)
            y = Dense(64, activation="linear")(y)
            y = LeakyReLU()(y)
            y = Dense(64, activation="linear")(y)
            y = LeakyReLU()(y)
            y = Dense(64, activation="linear")(y)
            y = Dense(1, activation="sigmoid")(y)

            model = Model(inputs=[ohlc_left_input,ohlc_middle_input,ohlc_right_input], outputs=y)
            return model

        return UpscalerModel(actor=create_actor(), critic=create_critic())

    def load_model(model_name):
        with open("{}.gan.pickle".format(model_name), mode="rb") as fd:
            memento = pickle.load(fd)
            return UpscalerModel(memento["generator"], memento["critic"])

    def exists(model_name):
        return os.path.isfile("{}.gan.pickle".format(model_name))

    def save_model(self, model_name):        
        memento = {
            "generator" : self.__generator,
            "critic" : self.__critic,
            "metadata" : "test"
        }
        fpath = "{}.gan.pickle".format(model_name)
        with open(fpath, mode="wb") as fd:
            pickle.dump(memento, fd)
        return fpath
        

    def generate_output(self, x):      
        assert len(x) == 3, "len(x) is {}. Should be 3".format(len(x))  
        return self.__generator.predict(x)

    def train_critic(self, real_samples_x, fake_samples_x, generator_critic_advantage):
        epochs = 2 if generator_critic_advantage >= 0.6 else 1

        for _ in range(epochs):
            # Train the critic to classify real samples
            real_samples_y = np.ones((real_samples_x[0].shape[0],1))
            real_samples_y = real_samples_y + np.random.normal(size=real_samples_y.shape, scale=0.1)
            loss_a = self.__critic_trainer.train_on_batch(real_samples_x, real_samples_y)

            # Train the critic to classify the "fake" samples
            fake_samples_y = np.zeros((fake_samples_x[0].shape[0],1))
            fake_samples_y = fake_samples_y + np.random.normal(size=fake_samples_y.shape, scale=0.1)
            loss_b = self.__critic_trainer.train_on_batch(fake_samples_x, fake_samples_y)
       
        return (loss_a, loss_b)

    def train_critic_invalid(self, invalid_samples_x):
        # Train the critic to classify the invalid samples
        invalid_samples_y = np.zeros((invalid_samples_x[0].shape[0],1))
        for _ in range(6):
            loss = self.__critic_trainer.train_on_batch(invalid_samples_x, invalid_samples_y)
        return loss

    def train_generator(self, x, critic_generator_advantage):
        # Train the generator to maximize the probability of being misclassified as "real" through the adverserial optimizer     
        epochs = 2 if critic_generator_advantage >= 0.6 else 1
        for _ in range(epochs):
            loss = self.__adverserial_trainer.train_on_batch(x, 0.9*np.ones((len(x[0]),1)) )
        return loss