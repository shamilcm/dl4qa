import argparse
import logging
from anssel import models
from anssel.data import Dataset, Embeddings
from anssel import utils
import os, sys

# Main method. The arguments to the training script!
if __name__=="__main__":
    parser = argparse.ArgumentParser();
    parser.add_argument("-ts", "--testset-path", required=True, dest="test_fname", help="test file")
    parser.add_argument("-tsref", "--testset-ref", required=True, dest="test_ref_fname", help="test reference file")
    parser.add_argument("-w", "--weights-file", required=True, dest="weights_fname", help="path/name of model. ")
    parser.add_argument("-emb", "--emb-path", required=True, dest="w2v_fname", help="path/name of pretrained word embeddings (Word2Vec inary). ")
    parser.add_argument( "-o", "--out-file", required=False, dest="out_fname", help="path to output rank file.. ")
    parser.add_argument("-s", "--system", dest="system", default="bow", help="bow | bigram | compdecomp")
    parser.add_argument("-d", "--device", dest="device", default="gpu", help="The computing device (cpu or gpu). Default: gpu")
    args = parser.parse_args()

logger = logging.getLogger(__name__)
utils.set_logger(None)

## Setting devie to use 
os.environ['THEANO_FLAGS'] = 'device=' + args.device


###########################
# Loading datasets

logger.info("Loading test set")
testset = Dataset(args.test_fname)


#############################
# Loading word embeddings
logger.info("Loading word embeddings")
embeddings = Embeddings(args.w2v_fname, testset.vocab)


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
testset.process(embeddings, stop_words)


logger.info("Building model")
hyperparams = models.HyperParams(emb_dim=embeddings.emb_dim)
binbowdense = models.BinaryBoWDense(hyperparams=hyperparams)

logger.info("Loading weights")
binbowdense.load_weights(args.weights_fname)
probs = binbowdense.predict(testset.samples)

logger.info("Evaluation")
from anssel import evaluator

preds = evaluator.get_preds(ref_file=args.test_ref_fname, probs=probs, out_file=args.out_fname)
logger.info("MAP:" + str(evaluator.calc_mean_avg_prec(preds)))
logger.info("MRR:" + str(evaluator.calc_mean_reciprocal_rank(preds)))
