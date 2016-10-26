import logging
import numpy as np
np.random.seed(1337)
from keras.models import Sequential, Model
from keras.layers import Dense,Activation,Merge,Input, Lambda,merge, Convolution1D, MaxPooling1D, Masking, Reshape
from keras.optimizers import SGD
import keras.backend as K
import sys
from scipy.spatial.distance import cdist

from anssel import utils
logger = logging.getLogger(__name__)

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


class BaseSystem:
   
    def save_model(self, out_file):
        """
        Save the model to a file
        """
        self.model.save_weights(out_file, overwrite='True')

    def load_weights(self, weights_file):
        self.model.load_weights(weights_file)


    def train_model(self, input, labels, dev_evaluator=None):
        self.model.fit(input, labels, nb_epoch=self.hyperparams.num_epochs, batch_size=self.hyperparams.batch_size)

    def train_model_by_epoch(self, input, labels, dev_evaluator=None):
        self.model.fit(input, labels, nb_epoch=1, batch_size=self.hyperparams.batch_size, verbose=0)



    def predict(self, input):
        probs = self.model.predict(input, batch_size=self.hyperparams.batch_size)
        return probs




# Yu et al. (2014) bag-of-words model for binary classification
class BinaryBoWDense(BaseSystem):
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

    def get_input(self, samples):
        qinputs_list = []
        ainputs_list = []
        qsamples = samples[0]
        asamples = samples[1]
        for ques_vecs in qsamples:
            qinputs_list.append(np.sum(ques_vecs, axis=0))
        for ans_vecs in asamples:
            ainputs_list.append(np.sum(ans_vecs, axis=0))
        return [np.array(qinputs_list), np.array(ainputs_list)]
      

    def get_labels(self, labels):
        logger.info("Input : ")
        logger.info(np.array(labels))
        return np.array(labels)

# Wang et al (2016) bag-of-words model for binary classification
class Wang2016CNN(BaseSystem):
    def __init__(self, hyperparams):
        from my_layers import MeanOverTime, MulConstant, Conv1DWithMasking
        
        logging.info("Initializing Wang2016CNN Model")
        self.hyperparams = hyperparams
        
        
        logger.error(hyperparams.emb_dim)
        qplus_input = Input(shape=(None,300),dtype='float32')
        qminus_input = Input(shape=(None,300),dtype='float32')
        aplus_input = Input(shape=(None,300),dtype='float32')
        aminus_input = Input(shape=(None,300),dtype='float32')
        
        # qplus_input_reshape = Reshape((1,3))(qplus_input)
        # qminus_input_reshape = Reshape((1,3))(qminus_input)
        # aplus_input_reshape = Reshape((1,3))(aplus_input)
        # aminus_input_reshape = Reshape((1,3))(aminus_input)
        
        # qplus_input_masking = Masking(mask_value=999)(qplus_input)
        # qminus_input_masking = Masking(mask_value=999)(qminus_input)
        # aplus_input_masking = Masking(mask_value=999)(aplus_input)
        # aminus_input_masking = Masking(mask_value=999)(aminus_input)
        
        cnn_border_mode = 'same'
        convQplus1 = Convolution1D(nb_filter=500, filter_length=1, border_mode=cnn_border_mode, subsample_length=1)(qplus_input)
        convQminus1 = Convolution1D(nb_filter=500, filter_length=1, border_mode=cnn_border_mode, subsample_length=1)(qminus_input)
        convAplus1 = Convolution1D(nb_filter=500, filter_length=1, border_mode=cnn_border_mode, subsample_length=1)(aplus_input)
        convAminus1 = Convolution1D(nb_filter=500, filter_length=1, border_mode=cnn_border_mode, subsample_length=1)(aminus_input)

        mergedQ1 = merge([convQplus1, convQminus1], mode='sum', concat_axis=-1)
        mergedQtanh = Activation('tanh')(mergedQ1)
        maxPoolQ = MeanOverTime()(mergedQtanh)
        
        mergedA1 = merge([convAplus1, convAminus1], mode='sum', concat_axis=-1)
        mergedAtanh = Activation('tanh')(mergedA1)
        maxPoolA = MeanOverTime()(mergedAtanh)
        
        combinedOutput = merge([maxPoolQ, maxPoolA], mode='concat', concat_axis=-1)
        
        
        # combinedOutput = merge([qplus_input, qminus_input], mode='concat', concat_axis=-1)
        densed = Dense(1)(combinedOutput)
        score = Activation('sigmoid')(densed)
        self.model = Model(input=[qplus_input,qminus_input,aplus_input,aminus_input], output=score)
        
        loss = 'binary_crossentropy'
        metric = 'accuracy'
        optimizer = SGD(lr=self.hyperparams.learning_rate, momentum=self.hyperparams.momentum, nesterov=True)
        
        self.model.compile(loss=loss, optimizer=optimizer, metrics=[metric])
        
    def get_input(self, samples):
        qinputs_list = []
        ainputs_list = []
        qsamples = samples[0]
        asamples = samples[1]
        q_maxlen = max(qsample.shape[0] for qsample in qsamples)
        a_maxlen = max(asample.shape[0] for asample in asamples)
        logger.info("q_maxlen : %i" % q_maxlen)
        logger.info("a_maxlen : %i" % a_maxlen)
        logger.info("Processing input for Wang et al.")
        maxlen_q = -1
        qplus_list = []
        qminus_list = []
        aplus_list = []
        aminus_list = []
        for qmatrix, amatrix in zip(qsamples, asamples):
            qplus, qminus, aplus, aminus = self.compose_decompose(qmatrix, amatrix)
            # Padding questions
            qpad_width = ((0,q_maxlen - qplus.shape[0]),(0,0))
            qplus_pad  = np.pad(qplus,  pad_width=qpad_width, mode='constant', constant_values=999)
            qminus_pad = np.pad(qminus, pad_width=qpad_width, mode='constant', constant_values=999)
            # Padding answers
            apad_width = ((0,a_maxlen - aplus.shape[0]),(0,0))
            aplus_pad  = np.pad(aplus,  pad_width=apad_width, mode='constant', constant_values=999)
            aminus_pad = np.pad(aminus, pad_width=apad_width, mode='constant', constant_values=999)
            # Adding these padded matrices to list
            qplus_list.append(qplus_pad)
            qminus_list.append(qminus_pad)
            aplus_list.append(aplus_pad)
            aminus_list.append(aminus_pad)
        logging.info("Converting list of matrices to 3-d tensor")
        qplus_tensor = np.array(qplus_list)
        qminus_tensor = np.array(qminus_list)
        aplus_tensor = np.array(aplus_list)
        aminus_tensor = np.array(aminus_list)
        #print qplus_tensor.shape
        #print qminus_tensor.shape
        #print aplus_tensor.shape
        #print aminus_tensor.shape

        logging.info("Processing complete!")
        #for ques_vecs in qsamples:
        #    qinputs_list.append(np.sum(ques_vecs, axis=0))
        #for ans_vecs in asamples:
        #    ainputs_list.append(np.sum(ans_vecs, axis=0))
        
        logger.info("qplus_tensor : ")
        # logger.info(qplus_tensor)
        logger.info(qplus_tensor.shape)
        logger.info("qminus_tensor : ")
        # logger.info(qminus_tensor)
        logger.info(qminus_tensor.shape)
        logger.info("aplus_tensor : ")
        # logger.info(aplus_tensor)
        logger.info(aplus_tensor.shape)
        logger.info("aminus_tensor : ")
        # logger.info(aminus_tensor)
        logger.info(aminus_tensor.shape)
        return [qplus_tensor, qminus_tensor, aplus_tensor, aminus_tensor]

    
    def compose_decompose(self, qmatrix, amatrix):
        qhatmatrix, ahatmatrix = self.f_match(qmatrix, amatrix, window_size=3)
        qplus, qminus = self.f_decompose(qmatrix, qhatmatrix)
        aplus, aminus = self.f_decompose(amatrix, ahatmatrix)
        return qplus, qminus, aplus, aminus
    
    def f_match(self, qmatrix, amatrix, window_size=3):    
        A = 1 - cdist(qmatrix, amatrix, metric='cosine') # Similarity matrix
        Atranspose = np.transpose(A)
        qa_max_indices = np.argmax(A, axis=1)        # 1-d array: for each question word, the index of the answer word which is most similar
        # Selecting answer word vectors in a window surrounding the most closest answer word  
        qa_window = [range( max(0,max_idx-window_size) , min(amatrix.shape[0], max_idx+window_size+1) ) for max_idx in qa_max_indices]
        # Selecting question word vectors in a window surrounding the most closest answer word
        # Finding weights and its sum (for normalization) to find f_match for question for the corresponding window of answer words
        qa_weights = [ (np.sum(A[qword_idx][aword_indices]), A[qword_idx][aword_indices]) for qword_idx, aword_indices in enumerate(qa_window)]
        # Then multiply each vector in the window with the weights, sum up the vectors and normalize it with the sum of weights
        # This will give the local-w vecotrs for the Question sentence words and Answer sentence words.       
        qhatmatrix = np.array([ np.sum(weights.reshape(-1,1) * amatrix[aword_indices], axis=0)/weight_sum for ((qword_idx, aword_indices),(weight_sum, weights))  in zip(enumerate(qa_window), qa_weights)])

        # Doing similar stuff for answer words
        aq_max_indices = np.argmax(A, axis=0)        # 1-d array: for each   answer word, the index of the question word which is most similar
        aq_window = [range( max(0,max_idx-window_size) , min(qmatrix.shape[0], max_idx+window_size+1) ) for max_idx in aq_max_indices]
        aq_weights = [ (np.sum(Atranspose[aword_idx][qword_indices]), Atranspose[aword_idx][qword_indices]) for aword_idx, qword_indices in enumerate(aq_window)]
        ahatmatrix = np.array([ np.sum(weights.reshape(-1,1) * qmatrix[qword_indices], axis=0)/weight_sum for ((aword_idx, qword_indices),(weight_sum, weights))  in zip(enumerate(aq_window), aq_weights)])
        return qhatmatrix, ahatmatrix
    
    def f_decompose(self, matrix, hatmatrix):
        # finding magnitude of parallel vector
        mag = np.sum(hatmatrix*matrix, axis=1)/np.sum(hatmatrix*hatmatrix, axis=1)
        # multiplying magnitude with hatmatrix vector
        plus = mag.reshape(-1,1)*hatmatrix
        minus = matrix - plus
        return plus, minus


    def get_labels(self, labels):
        return np.array(labels)
