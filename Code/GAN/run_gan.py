import signal
import sys

import keras

from gan import MNIST_DCGAN, ElapsedTimer, generator_model_path, discriminator_model_path

if 'load' in sys.argv:
    print("Loading generator model from: ", generator_model_path)
    generator = keras.models.load_model(generator_model_path)
    print("Loading discriminator model from: ", discriminator_model_path)
    discriminator = keras.models.load_model(discriminator_model_path)
    print("Initializing DCGAN")
    dcgan = MNIST_DCGAN(discriminator=discriminator, generator=generator)
else:
    dcgan = MNIST_DCGAN()


# define train INTERRUPT signal handler,
# which will save model before exiting the program
def signal_handler(sig, frame):
    print("signal_handler: making terminate request")
    # make training stop
    dcgan.terminate = True


# setup INTERRUPT signal handler
signal.signal(signal.SIGINT, signal_handler)

# run the DCGAN
timer = ElapsedTimer()
dcgan.train(train_steps=10000, batch_size=256, plot_interval=500)
timer.elapsed_time()
dcgan.plot_images(fake=True, save2file=True)
dcgan.plot_images(fake=False, save2file=True)
