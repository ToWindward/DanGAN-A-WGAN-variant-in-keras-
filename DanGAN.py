
import numpy as np
from keras.layers import Input, Dense,Conv2D,Flatten, Dropout, Reshape,Conv2DTranspose, BatchNormalization, LeakyReLU
import keras.backend as K
from keras.optimizers import RMSprop, Adam, SGD, Adadelta
from keras.models import Model
from keras.initializers import RandomNormal
from keras.constraints import max_norm
from keras.models import load_model

#creating and initialising the class, this includes quite a lot of variable to create optional diagnostic charts. Images for training need to be passed on initialisation.
class DanGAN():
    def __init__(self,images):
        self.testout = 0
        self.gen, self.dis, self.combo = self.build_models()
        self.fakes = []
        self.glos = []
        self.dlos = []
        self.clip_value=0.02
        self.l1 = []
        self.l2 = []
        self.testval = []
        self.prefake = []
        self.postfake = []
        self.diff = []
        self.current = 0
        self.images = images
        self.diff2 = []
        self.learn_dis = []
        self.learn_combo = []
        self.disclip = []
        self.genloss = []
        self.genit = []
        self.disit = []
        self.testvar = []
        self.postvar = []
        self.prevar = []
        self.comboloss = []

    #This is where the generator and disriminator are built
    def build_models(self):
        
        #optimiser for the discriminator
        op1 = RMSprop(0.002)
        
        #optimiser for the generator
        op2 = RMSprop(lr=0.001,decay=0.001)

        #loss function for the discriminator, the standard deviation term penalises the discriminator for assigning wildly varying values for imgaes within a given real or fake batch.
        def ganloss(y_true,y_pred):
            return K.mean(y_true*y_pred) + 0.1*K.std(y_pred)


        #loss function for the generator, aims to minimise the difference between the mean and standard deviation of the generated images compared with the real images (both terms are squared)
        def comboloss(y_true,y_pred):
            loss = K.square(K.mean(y_pred) - K.mean(y_true)) +  K.square(K.std(y_pred) - K.std(y_true))
            return loss

        #input for the images
        realim = Input(shape=(128,128,3))
        
        #noise vector input
        vect = Input(shape=(100,))
 
        #This is the disciminator network
        discriminator = Conv2D(32,kernel_size=7,strides=2,padding='same',kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None),
                               kernel_constraint=max_norm(10,axis=[0,1,2]), bias_constraint=max_norm(10,axis=[0]))(realim)
        discriminator = LeakyReLU(alpha=0.2)(discriminator)
        discriminator = BatchNormalization(momentum=0.8)(discriminator)
        discriminator = Dropout(rate=0.2)(discriminator)



        discriminator = Conv2D(48, kernel_size=5, strides=2, padding="same",
                               kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None),
                               kernel_constraint=max_norm(10, axis=[0, 1, 2]),
                               bias_constraint=max_norm(10, axis=[0]))(discriminator)
        discriminator = LeakyReLU(alpha=0.2)(discriminator)
        discriminator = BatchNormalization(momentum=0.8)(discriminator)

        discriminator = Dropout(rate=0.2)(discriminator)

        discriminator = Conv2D(64, kernel_size=5, strides=2, padding="same",
                               kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None),
                               kernel_constraint=max_norm(10, axis=[0, 1, 2]),
                               bias_constraint=max_norm(10, axis=[0]))(discriminator)
        discriminator = LeakyReLU(alpha=0.2)(discriminator)
        discriminator = BatchNormalization(momentum=0.8)(discriminator)

        discriminator = Dropout(rate=0.2)(discriminator)

        discriminator = Conv2D(96, kernel_size=5, strides=2, padding="same",
                               kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None),
                               kernel_constraint=max_norm(10, axis=[0, 1, 2]),
                               bias_constraint=max_norm(10, axis=[0]))(discriminator)
        discriminator = LeakyReLU(alpha=0.2)(discriminator)
        discriminator = BatchNormalization(momentum=0.8)(discriminator)

        discriminator = Dropout(rate=0.2)(discriminator)

        discriminator = Conv2D(128, kernel_size=5, strides=2, padding="same",
                               kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None),
                               kernel_constraint=max_norm(10, axis=[0, 1, 2]),
                               bias_constraint=max_norm(10, axis=[0]))(discriminator)
        discriminator = LeakyReLU(alpha=0.2)(discriminator)
        discriminator = BatchNormalization(momentum=0.8)(discriminator)

        discriminator = Dropout(rate=0.2)(discriminator)

        discriminator = Conv2D(256, kernel_size=3, strides=1, padding='same',kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None),
                               kernel_constraint=max_norm(10,axis=[0,1,2]), bias_constraint=max_norm(10,axis=[0]))(discriminator)
        discriminator = LeakyReLU(alpha=0.2)(discriminator)
        discriminator = BatchNormalization(momentum=0.8)(discriminator)
 
        discriminator = Dropout(rate=0.2)(discriminator)

        discriminator = Flatten()(discriminator)
        discriminator = Dense(1,kernel_constraint=max_norm(10), bias_constraint=max_norm(10), activation='linear')(discriminator)


        disc = Model(inputs=[realim],outputs=[discriminator])
        disc.compile(optimizer=op1,
                       loss=[ganloss])


        #this is the generator network
        
        generator = Dense(512 * 2 * 2)(vect)
        generator = LeakyReLU(alpha=0.2)(generator)
        generator = BatchNormalization(momentum=0.8)(generator)
   
        generator = Reshape((2, 2, 512))(generator)
        generator = Dropout(0.2)(generator)

        generator = Conv2DTranspose(256, kernel_size=3,strides=2, padding="same",kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None))(generator)
        generator = LeakyReLU(alpha=0.2)(generator)
        generator = BatchNormalization(momentum=0.8)(generator)
        generator = Dropout(0.2)(generator)

        generator = Conv2DTranspose(128, kernel_size=3, strides=2, padding="same")(generator)
        generator = LeakyReLU(alpha=0.2)(generator)
        generator = BatchNormalization(momentum=0.8)(generator)
        generator = Dropout(0.2)(generator)

        generator = Conv2DTranspose(96, kernel_size=3, strides=2, padding="same",kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None))(generator)
        generator = LeakyReLU(alpha=0.2)(generator)
        generator = BatchNormalization(momentum=0.8)(generator)
        generator = Dropout(0.2)(generator)

        generator = Conv2DTranspose(64, kernel_size=5, strides=2, padding="same",kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None))(generator)
        generator = LeakyReLU(alpha=0.2)(generator)
        generator = BatchNormalization(momentum=0.8)(generator)
        generator = Dropout(0.2)(generator)

        generator = Conv2DTranspose(48, kernel_size=5, strides=2, padding="same",kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None))(generator)
        generator = LeakyReLU(alpha=0.2)(generator)
        generator = BatchNormalization(momentum=0.8)(generator)
        generator = Dropout(0.2)(generator)

        generator = Conv2DTranspose(32, kernel_size=7, strides=2, padding="same",kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None))(generator)
        generator = LeakyReLU(alpha=0.2)(generator)
        generator = BatchNormalization(momentum=0.8)(generator)
        generator = Dropout(0.2)(generator)

        generator = Conv2DTranspose(3, kernel_size=9, strides=1, padding="same",kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None),activation='tanh',)(generator)
  

        gen = Model([vect],[generator])
        
        #this is the combined discriminator and generator network (the generator itself is not compiled). The weights of the discriminator are fixed to ensure that it is static when the generator is trained.
        discriminator_fixed = Model(inputs=[realim], outputs=discriminator)
        discriminator_fixed.trainable = False

        combo = discriminator_fixed(generator)

        comb = Model([vect],[combo])

        comb.compile(optimizer=op2,
                      loss=[comboloss])

        return gen,disc,comb


    #Code for training the networks, running this a single time will run 1 training loop for the discriminator and generator
    def train(self):

        batch_size=64

        #creating the noise vector
        vectors = np.random.normal(0,1,(batch_size,100))
        
        #predicting fake images
        imfk = self.gen.predict(vectors)

        self.current = imfk

        self.fakes.append(imfk[0])
        
        #selecting a random batch of images
        idx = np.random.randint(len(self.images), size=batch_size)
        imsam = self.images[idx]
        imsam = (imsam.astype(np.float32) - 127.5) / 127.5
        
        #numpy arrays for true and fake image training (negative ones to encourage positive discriminator outputs, positive to encourage negative discriminator outputs)
        true = -np.ones(batch_size)
        false = np.ones(batch_size)


        #calculate the current value of the discriminator outputs for real and fake images
        self.dlos1 = -np.mean(self.dis.predict(imsam))
        self.dlos2 = np.mean(self.dis.predict(imfk))
 
        #This loop trains the disciminator on real and fake images until either the outputs are of opposite sign and out1/out2 <0.4, or until 20 iterations have passed. These parameters are tunable. 
        # At each iteration the learning rate is divided by the square of the output (this requires extra predict steps which do add a slight time penalty to the code)  
        i = 0
        while (((self.dlos1 > 0) or (self.dlos2 > 0) or (abs(self.dlos2 / self.dlos1) < 0.4) or (
                abs(self.dlos1 / self.dlos2) < 0.4)) or (i<3)) and (i<20):
            i += 1
            if self.dlos1 > 0:
                K.set_value(self.dis.optimizer.lr,
                             0.001)
            else:
                K.set_value(self.dis.optimizer.lr,
                            min(1 / (self.dlos1 * self.dlos1), 0.0001))

            x1 = self.dis.train_on_batch(imsam, true)

            self.dlos2 = np.mean(self.dis.predict(imfk))
 
            if self.dlos2 > 0:
                K.set_value(self.dis.optimizer.lr,
                            0.001)
            else:
                K.set_value(self.dis.optimizer.lr,
                            min(1 / (self.dlos2 * self.dlos2), 0.0001))
            x2 = self.dis.train_on_batch(imfk, false)

            self.dlos1 = -np.mean(self.dis.predict(imsam))
            self.dlos2 = np.mean(self.dis.predict(imfk))

        #Storing some outputs for charts
        self.dlos.append(x1)
        self.glos.append(x2)
        print('dis : ', i)
        self.disit.append(i)
        tval = self.dis.predict(imsam)
        pval = self.combo.predict(vectors)
        self.testout = np.var(tval)


        #Code for training the generator. Generator is trained until the difference between the generator and discriminator mean is less than 20% of the discriminator mean. 
        # Or until 10 iterations have passed, this is necessary, because training the generator too much on a single batch can lead to overfitting.
        genloss = -np.mean(pval)
        i = 0 
        while ((abs(genloss - np.mean(tval)) > (0.2*np.mean(tval))) and (i < 10)) or (i < 3):
            i += 1
 
            x3 =  self.combo.train_on_batch(vectors, tval)
            self.comboloss.append(x3)
            genloss = -np.mean(self.combo.predict(vectors))
        self.genloss.append(genloss)
   
        #Adding some more data for charts
        print('gen : ', i)
        self.genit.append(i)

        vals = self.combo.predict(vectors)
 
        self.postfake.append(np.mean(vals))
        self.prefake.append(np.mean(pval))
        self.testval.append(np.mean(tval))
        self.postvar.append(np.var(vals))
        self.prevar.append(np.var(pval))
        self.testvar.append(np.var(tval))
        self.diff.append(np.mean(tval-pval))
        self.diff2.append(np.mean(vals - pval))
   
    #saving and loading the models
    def save_models(self):
        self.dis.save('dis.h5')
        self.gen.save('gen.h5')
        self.combo.save('gen.h5')

    def del_models(self):
        del self.dis
        del self.gen
        del self.combo

    def load_models(self):
        self.dis = load_model('dis.h5')
        self.gen = load_model('dis.h5')
        self.combo = load_model('dis.h5')


