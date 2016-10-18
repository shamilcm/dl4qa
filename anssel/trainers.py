import logging

class BinaryTrainer:
	def __init__(self, train_file, dev_file, emb_file, epoch, batchsize):
		logging.info("Initializing trainer")

