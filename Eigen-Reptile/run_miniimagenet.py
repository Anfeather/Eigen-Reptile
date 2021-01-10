"""
Train a model on miniImageNet.
"""

import random

import tensorflow as tf

from IER.args import argument_parser, model_kwargs, train_kwargs, evaluate_kwargs
from IER.eval import evaluate
from IER.models import MiniImageNetModel
from IER.miniimagenet import read_dataset
from IER.train import train
import os
import sys
current_path = os.getcwd()
sys.path.append(current_path)

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
DATA_DIR = "/home/ray/preject/eig_reptile/supervised-reptile-master/data/mini-imagenet"

def main():
    """
    Load data and train a model on it.
    """
    args = argument_parser().parse_args()
    random.seed(args.seed)

    train_set, val_set, test_set = read_dataset(DATA_DIR)
    model = MiniImageNetModel(args.classes, **model_kwargs(args))

    with tf.Session() as sess:
        if not args.pretrained:
            print('Training...')
            train(sess, model, train_set, test_set, args.checkpoint, **train_kwargs(args))

        print('Evaluating...')
        eval_kwargs = evaluate_kwargs(args)
        print('Train accuracy: ' + str(evaluate(sess, model, train_set, **eval_kwargs)))
        print('Validation accuracy: ' + str(evaluate(sess, model, val_set, **eval_kwargs)))
        print('Test accuracy: ' + str(evaluate(sess, model, test_set, **eval_kwargs)))

if __name__ == '__main__':
    main()
