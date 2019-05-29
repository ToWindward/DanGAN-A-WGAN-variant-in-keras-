
import numpy as np
from DanGAN import DanGAN
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

#importing images, you need to modify this for wherever your images are saved.
imlst = []
for a in range(1,1333):
    try:
        im = Image.open('faces/f ('+str(a)+').jpg')
        im = np.array(im)
        im = cv2.resize(im,(128,128))
        imlst.append(im)
    except:
        print('skipped ', str(a))

images = np.array(imlst)

#bonus list for moving average chart
diff2_trail = []

#create instance of the class
gan = DanGAN(images)

#print a summary of the networks
gan.combo.summary()

#There are 2 loops here, every time the inner loop completes the models are saved and a batch of images are saved, this is to save space and in case of crashes etc
# This can be done however you want but having 2 loops makes it easy to adjust.
for b in range(100):

    iterations = 64
    for a in range(iterations):
        #run the training function
        gan.train()

        #code for creating optional charts
        if a%8==0:
            plt.plot(gan.diff)
            plt.ylim(0)

            plt.savefig('diff')
            plt.close()

            plt.plot(gan.diff2)

            plt.savefig('diff2')
            plt.close()
            diff2_trail.append(np.mean(np.array(gan.diff2[-8:])))

            plt.plot(diff2_trail)
            plt.ylim(0)
            plt.savefig('diff2_trail')
            plt.close()

            plt.plot(gan.testval)
            plt.plot(gan.prefake)
            plt.plot(gan.postfake)
            plt.savefig('true and fake pred')
            plt.close()

            plt.plot(gan.testvar)
            plt.plot(gan.prevar)
            plt.plot(gan.postvar)
            plt.savefig('true and fake var')
            plt.close()


            plt.plot(gan.learn_dis)
            plt.plot(gan.learn_combo)

            plt.ylim(0)
            plt.savefig('learn')
            plt.close()

            plt.plot(gan.disclip)

            plt.savefig('dis_clipnorm')
            plt.close()

            plt.plot(-np.array(gan.dlos))

            plt.plot(gan.glos)
            plt.plot(-np.array(gan.genloss))

            plt.savefig('loss')
            plt.close()



            plt.plot(gan.disit)
            plt.plot(gan.genit)

            plt.savefig('iterations')
            plt.close()

            plt.plot(gan.comboloss)
            plt.savefig('comboloss')
            plt.close()

        #saving the models  (I removed loading because it doesn't currently work and I havent needed it but I do plan to add it back in)
        gan.save_models()


    #Code to save a batch of images, happens every time the 
    def show_images(images, cols=1, titles=None, name='jeff.png'):

        assert ((titles is None) or (len(images) == len(titles)))
        n_images = len(images)
        if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
        fig = plt.figure()
        for n, (image, title) in enumerate(zip(images, titles)):
           a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
           if image.ndim == 2:
               plt.gray()
           plt.imshow(image)
        fig.set_size_inches(16,16)
        plt.axis('off')
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(name, bbox_inches='tight')
        plt.close()


    # change the output from -1:1 to 0-255
    gcurr = ((gan.current+1)*127.5).astype(int)
    gfake = ((np.array(gan.fakes)+1)*127.5).astype(int)
    show_images(list(gcurr),8,name='./face/rms_middisbiggen_decay_128_'+ str(b)+'.png')
