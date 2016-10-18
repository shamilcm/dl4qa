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

    def test_model(self, input, labels):
        probs = self.model.predict(input, batch_size=self.hyperparams.batch_size)
        correct_1 = 0
        total_1=0
        correct_0=0
        total_0=0
        cut_off=0.5
        for i in range(0, len(probs)):
            if labels[i]==1:
                total_1 +=1
            if probs[i] >= cut_off and labels[i] == 1:
                correct_1 += 1
            if  labels[i]==0:
                total_0 +=1
            if probs[i] < cut_off and labels[i] == 0:
                correct_0 += 1
        print "Accuracy 1: ", float(correct_1) / total_1
        print "Accuracy 0: ", float(correct_0) / total_0
        print "Avg. Accuracy: ", (float(correct_1) / total_1 + float(correct_0) / total_0)/2

        print "Finished......"

