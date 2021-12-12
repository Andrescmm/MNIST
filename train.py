import numpy as np                   # advanced math library
import matplotlib.pyplot as plt      # MATLAB like plotting routines
import random                        # for generating random numbers
import tensorflow as tf
from keras.datasets import mnist
import sys
import math



(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = tf.keras.utils.normalize(X_train, axis=1)
X_test = tf.keras.utils.normalize(X_test, axis=1)

print("X_train shape", X_train.shape)
print("y_train shape", y_train.shape)
print("X_test shape", X_test.shape)
print("y_test shape", y_test.shape)


# now print!
orig_stdout = sys.stdout
f=open('pVectors.txt', 'w')
sys.stdout = f
for i in range(60000):
  data = X_train[i].reshape(-1).tolist()
  print([math.ceil(data) for data in data])
  print('\n')
sys.stdout = orig_stdout
f.close()
