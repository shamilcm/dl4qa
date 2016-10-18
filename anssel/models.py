import logging
from keras.models import Sequential, Model
from keras.layers import Dense,Activation,Merge,Input, Lambda,merge
from keras.optimizers import SGD
import keras.backend as K


### Hyper Parameters Class
class HyperParams:
    def __init__(self, num_epochs=20, batch_size=100, emb_dim=300):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.emb_dim = emb_dim
        self.learning_rate = 0.001
        self.momentum = 0.9



############################
##  Models
############################

# Yu et al. (2014) bag-of-words model for binary classification
class BinaryBoWDense:
    def __init__(self, hyperparams):
        logging.info("Initializing Binary Bag-of-words Model")
        self.hyperparams = hyperparams
        s1_input = Input(shape=(hyperparams.emb_dim,),dtype='float32',name='s1_input')
        s2_input = Input(shape=(hyperparams.emb_dim,),dtype='float32',name='s2_input')
        s1_dense = Dense(output_dim=hyperparams.emb_dim,input_dim=hyperparams.emb_dim,activation='linear')(s1_input)
        s1_s2_merged = merge([s1_dense, s2_input], mode='dot', dot_axes=(1, 1))
        labels=Activation('sigmoid',name='labels')(s1_s2_merged)
        self.model=Model(input=[s1_input,s2_input],output=[labels])
        self.model.compile(loss={'labels':'binary_crossentropy'}, optimizer=SGD(lr=self.hyperparams.learning_rate, momentum=self.hyperparams.momentum, nesterov=True), metrics=['accuracy'])

    def train_model(self, input, labels):
        self.model.fit(input, labels, nb_epoch=self.hyperparams.num_epochs, batch_size=self.hyperparams.batch_size)

    def save_model(self, out_file):
        """
        Save the model to a file
        """
        self.model.save_weights(out_file, overwrite='True')

    def load_weights(self, weights_file):
        self.model.load_weights(weights_file)

    def predict(self, input):
        probs = self.model.predict(input, batch_size=self.hyperparams.batch_size)
        return probs
