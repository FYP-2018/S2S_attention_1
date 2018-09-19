import tensorflow as tf
import subprocess
import logging
import os

import sys
sys.path.insert(0, '/Users/user/PycharmProjects/Seq2Seq/TensorFlow-Summarization')
from src import data_util

MAX_STEPS = 300000
STEPS_PER_VALIDATION = 1000
STEPS_PER_CHECKPOINT = 20000
TEST_THRESHOLD = 200000

MAX_STEPS = 300
STEPS_PER_VALIDATION = 10
STEPS_PER_CHECKPOINT = 20
TEST_THRESHOLD = 200


train_params = {
    "--steps_per_validation": STEPS_PER_VALIDATION,
    "--steps_per_checkpoint": STEPS_PER_CHECKPOINT,
}

############################################################
# extract the data-loading procedure here
# so that dont need to load data for every training epoch
# @Crystina
############################################################

DATA_DIR = 'data'
DOC_VOCAB_SIZE = 30000
SUM_VOCAB_SIZE = 30000

global dataset
dataset = None

if dataset == None:
    print ('-- loading dataset')
    docid, sumid, doc_dict, sum_dict = \
        data_util.load_data(
            DATA_DIR + "/train.article.txt",
            DATA_DIR + "/train.title.txt",
            DATA_DIR + "/doc_dict.txt",
            DATA_DIR + "/sum_dict.txt",
            DOC_VOCAB_SIZE,
            SUM_VOCAB_SIZE)

    val_docid, val_sumid = \
        data_util.load_valid_data(
            DATA_DIR + "/valid.article.filter.txt",
            DATA_DIR + "/valid.title.filter.txt",
            doc_dict, sum_dict)

    dataset = [docid, sumid, doc_dict, sum_dict]

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s",
                        datefmt='%b %d %H:%M')

    try:
        global_step = tf.contrib.framework.load_variable("model", "global_step")

    except:
        global_step = 0

    logging.info("Training starts with global_step={}. ".format(global_step))

    # dataset = [1, 2, 3, 4]

    while global_step < MAX_STEPS:
        
        terminate_step = max(global_step + STEPS_PER_CHECKPOINT, TEST_THRESHOLD)

        logging.info("Train from {} to {}. ".format(global_step, terminate_step))

        train_proc = ["python3", "src/summarization.py", "--max_iter", str(terminate_step)]
        test_proc = ["python3",  "script/test.py"]

        for key, val in train_params.items():
            train_proc.append(key)
            train_proc.append(str(val))

        subprocess.call(train_proc)

        global_step = terminate_step

        # subprocess.call(["python3", "script/test.py"])
        subprocess.call(test_proc)