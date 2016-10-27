import argparse
import logging
from anssel.data import Dataset, Embeddings
import os
import sys
from anssel import utils
# Test#
# Main method. The arguments to the training script!
if __name__=="__main__":
    parser = argparse.ArgumentParser();
    parser.add_argument("-tr", "--trainset-path", dest="train_fname", required=True, help="train file")
    parser.add_argument("-dev", "--devset-path", dest="dev_fname", required=True, help="development file")
    parser.add_argument("-devr", "--devset-ref", required=True, dest="dev_ref_fname", help="test reference file")
    parser.add_argument("-test", "--testset-path", required=False, dest="test_fname", help="test file")
    parser.add_argument("-testr", "--testset-ref", required=False, dest="test_ref_fname", help="test reference file")
    parser.add_argument("-emb", "--emb-path", required=True, dest="w2v_fname", help="path/name of pretrained word embeddings (Word2Vec inary). ")
    parser.add_argument("-d", "--device", dest="device", default="gpu", help="The computing device (cpu or gpu). Default: gpu")
    parser.add_argument("-s", "--system", dest="system", default="bow", help="bow | bigram | compdecomp")
    parser.add_argument("-E", "--num-epochs", dest="num_epochs", default=20, type=int, help="Number of iterations (epochs). Default: 10")
    parser.add_argument("-B", "--batch-size", dest="batch_size", default=100, type=int, help="Minibatch size for training. Default: 100")
    parser.add_argument("-dir", "--output-directory", dest="out_dir", help="The output directory for log file, model, etc.")

    args = parser.parse_args()


utils.mkdir_p(args.out_dir)
logger = logging.getLogger(__name__)
utils.set_logger(args.out_dir)

## Setting devie to use 
os.environ['THEANO_FLAGS'] = 'device=' + args.device


###########################
# Loading datasets

logger.info("Loading datasets")
trainset = Dataset(args.train_fname)
devset = Dataset(args.dev_fname)
if(args.test_fname):
    testset = Dataset(args.test_fname)


#############################
# Loading word embeddings
logger.info("Loading word embeddings")
embeddings = Embeddings(args.w2v_fname, trainset.vocab)

#############################
# Processing datasets with word embeddings
logger.info("Processing datasets with word embeddings")
if args.system == "bow":
    curdir = os.path.dirname(os.path.abspath(__file__))
    stopwords_fname = os.path.join(curdir, 'resources/short-stopwords.txt')
    stop_words=dict()
    with open(stopwords_fname) as f:
        for line in f:
            line = line.strip()
            stop_words[line] = True
else:
    stop_words = None

trainset.process(embeddings, stop_words)
devset.process(embeddings, stop_words)
if(args.test_fname):
    testset.process(embeddings, stop_words)

logger.info("Building model")
from anssel import models
hyperparams = models.HyperParams(num_epochs=args.num_epochs, batch_size=args.batch_size, emb_dim=embeddings.emb_dim)

if args.system == "bow":
    system = models.BinaryBoWDense(hyperparams=hyperparams)
elif args.system == "compdecomp":
    system = models.Wang2016CNN(hyperparams=hyperparams)


logger.info("Training")
from anssel import evaluator
utils.mkdir_p(args.out_dir +'/models')
best_dev_map = 0
best_model = None
train_in = system.get_input(trainset.samples)
train_labels = system.get_labels(trainset.labels)
dev_in = system.get_input(devset.samples)
for epoch in xrange(system.hyperparams.num_epochs):
    logger.info("Epoch:" + str(epoch))
    system.train_model_by_epoch(train_in, train_labels)  # training for one epoch
    probs = system.predict(dev_in)
    preds = evaluator.get_preds(ref_file=args.dev_ref_fname, probs=probs)
    dev_map = evaluator.calc_mean_avg_prec(preds)
    dev_mrr = evaluator.calc_mean_reciprocal_rank(preds)
    logger.info("Devset MAP=" + str(dev_map) + ", MRR=" + str(dev_mrr))
    system.save_model(args.out_dir + '/models/model.epoch_' + str(epoch) + '.h5')


logger.info("Evaluation on test set")
if (args.test_fname):
    probs = system.predict(system.get_input(testset.samples))
    if args.test_ref_fname:
        preds = evaluator.get_preds(ref_file=args.test_ref_fname, probs=probs)
        logger.info("MAP:" + str(evaluator.calc_mean_avg_prec(preds)))
        logger.info("MRR:" + str(evaluator.calc_mean_reciprocal_rank(preds)))


