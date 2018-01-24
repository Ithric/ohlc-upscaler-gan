

class UpscalerModel(object):
    def __init__(self, actor, critic):
        self.__generator = actor
        self.__critic = critic
     
        self.__critic_trainer = UpscalerModel.__create_critic_trainer()
        self.__adverserial_trainer = UpscalerModel.__create_adverserial_trainer()

    def __create_critic_trainer(critic):
        from keras.optimizers import RMSprop
        
        optimizer = RMSprop(lr=0.0002, decay=6e-8)
        discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return discriminator

    def __create_adverserial_trainer(actor, critic):
        from keras.layers import Input
        from keras.optimizers import Adam
        
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
            middle = Flatten()(ohlc_middle_input)
            middle = Dense(128, activation="relu")(middle)
            middle = Dense(128, activation="relu")(middle)
            middle = Dense(128, activation="relu")(middle)
            middle = Dense(upscaling_factor*5, activation="linear")(middle)
            middle = Reshape(target_shape=(upscaling_factor,5))(middle)

            model = Model(inputs=[ohlc_left_input, ohlc_middle_input, ohlc_right_input], outputs=[ohlc_left,y,ohlc_right])
            return model

        def create_critic():
            """ map data:  (ohlc_t-1, ohlc_t0 @ upscaling_factor, ohlc_t+1) -> (real: yes/no)
                map shape: [(5,),(upscaling_factor,5),(5,)]                 -> (1,)
            """
            ohlc_left_input = Input(shape=(5,))
            ohlc_middle_input = Input(shape=(upscaling_factor,5))
            ohlc_right_input = Input(shape=(5,))

            ohlc_middle = Flatten()(ohlc_middle_input)
            y = concatenate([ohlc_left_input, ohlc_middle_input, ohlc_right_input])
            y = Dense(256, activation="relu")(y)
            y = Dense(128, activation="relu")(y)
            y = Dense(1, activation="sigmoid")(y)

            model = Model(inputs=[ohlc_left_input,ohlc_middle_input,ohlc_right_input], outputs=y)
            return model

        return UpscalerModel(actor=create_actor(), critic=create_critic())

    def load_model(model_name):
        with open("{}.gan.pickle".format(model_name), mode="wb") as fd:
            memento = pickle.load(fd)
            return UpscalerModel(memento["generator"], memento["critic"])

    def save_model(self, model_name):
        memento = {
            "generator" : self.__generator,
            "critic" : self.__critic,
            "metadata" : "test"
        }
        with open("{}.gan.pickle".format(model_name), mode="rb") as fd:
            pickle.dump(memento, fd)

    def generate_output(self, x, priors):      
        assert len(x) == 3, "len(x) is {}. Should be 3".format(len(x))  
        tmp = self.__generator.predict(x)
        print("generator=>",tmp)
        return tmp

    def train_critic(self, real_samples_x, fake_samples_x):
        # Train the critic to classify real samples
        real_samples_y = np.ones((len(real_samples_x),1))        
        loss_a = self.__critic_trainer.train_on_batch(real_samples)

        # Train the critic to classify the "fake" samples
        fake_samples_y = np.zeros((len(fake_samples_x),1))
        loss_b = self.__critic_trainer.train_on_batch(fake_samples)

        return (loss_a, loss_b)

    def train_generator(self, x, priors):        
        # Train the generator to maximize the probability of being misclassified as "real" through the adverserial optimizer
        loss = self.__adverserial_trainer.train_on_batch(x, np.ones((len(x),1)) )
        return loss