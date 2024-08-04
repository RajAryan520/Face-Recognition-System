# Custom l1 Distance layer module
#Reason: needed to load the custom model

# import dependencies
import tensorflow as tf
from keras.layers import Layer

# Custom L1 Distance Layer from Jupyter
class L1Dist(Layer):

    #Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()

    # similarity calculation
    def call(self,inputs):
        input_embedding,validation_embedding = inputs
        return tf.abs(input_embedding - validation_embedding)
