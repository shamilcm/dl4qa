import argparse
import logging
from anssel.data import Dataset, Embeddings
import os
import sys
from anssel import utils

# Main method. The arguments to the training script!
if __name__=="__main__":
    parser = argparse.ArgumentParser();
    parser.add_argument("-tr", "--trainset-path", dest="train_fname", required=True, help="train file")
    parser.add_argument("-tu", "--devset-path", dest="dev_fname", required=True, help="development file")
    parser.add_argument("-ts", "--testset-path", required=False, dest="test_fname", help="test file")
    parser.add_argument("-tsref", "--testset-ref", required=False, dest="test_ref_fname", help="test reference file")
    parser.add_argument("-emb", "--emb-path", required=True, dest="w2v_fname", help="path/name of pretrained word embeddings (Word2Vec inary). ")
    parser.add_argument("-d", "--device", dest="device", default="gpu", help="The computing device (cpu or gpu). Default: gpu")
    parser.add_argument("-E", "--num-epochs", dest="num_epochs", default=20, type=int, help="Number of iterations (epochs). Default: 20")
    parser.add_argument("-B", "--batch-size", dest="batch_size", default=100, type=int, help="Minibatch size for training. Default: 100")
    parser.add_argument("-o", "--output-directory", dest="out_dir", help="The output directory for log file, model, etc.")

    args = parser.parse_args()


utils.mkdir_p(args.out_dir)
logger = logging.getLogger(__name__)
utils.set_logger(args.out_dir)

## Setting devie to use 
os.environ['THEANO_FLAGS'] = 'device=' + args.device


###########################
# Loading datasets

logger.info("Loading training set")
trainset = Dataset(args.train_fname)
logger.info("Loading development set")
devset = Dataset(args.dev_fname)
if(args.test_fname):
    logger.info("Loading test set")
    testset = Dataset(args.test_fname)


#############################
# Loading word embeddings
logger.info("Loading word embeddings")
embeddings = Embeddings(args.w2v_fname, trainset.vocab)

#############################
# Processing datasets with word embeddings
logger.info("Processing datasets with word embeddings")
curdir = os.path.dirname(os.path.abspath(__file__))
stopwords_fname = os.path.join(curdir, 'resources/short-stopwords.txt')
stop_words=dict()
with open(stopwords_fname) as f:
    for line in f:
        line = line.strip()
        stop_words[line] = True

trainset.process(embeddings, stop_words)
devset.process(embeddings, stop_words)
if(args.test_fname):
    testset.process(embeddings, stop_words)

logger.info("Building model")
from anssel import models
hyperparams = models.HyperParams(num_epochs=args.num_epochs, batch_size=args.batch_size, emb_dim=embeddings.emb_dim)
binbowdense = models.BinaryBoWDense(hyperparams=hyperparams)

logger.info("Training")
binbowdense.train_model(trainset.samples, trainset.labels)

logger.info("Saving model")

utils.mkdir_p(args.out_dir +'/models')
num_epoch = binbowdense.hyperparams.num_epochs
binbowdense.save_model(args.out_dir + '/models/model.epoch_' + str(num_epoch) + '.h5')

logger.info("Testing")
if (args.test_fname):
    probs = binbowdense.predict(testset.samples)
    from anssel import evaluator
    evaluator.print_accuracy(probs, testset.labels)




