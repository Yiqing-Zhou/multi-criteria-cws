import argparse
import time
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-epochs", default=10, dest="num_epochs", type=int,
                        help="Number of full passes through training set")
    parser.add_argument("--dropout", default=0.5, dest="dropout", type=float,
                        help="Amount of dropout(not keep rate, but drop rate) to apply to embeddings part of graph")
    parser.add_argument("--hidden-dim", default=128, dest="hidden_dim", type=int, help="Size of LSTM hidden layers")
    parser.add_argument("--learning-rate", default=0.01, dest="learning_rate", type=float, help="Initial learning rate")
    parser.add_argument("--learning-rate-decay", default=1e-4, dest="learning_rate_decay", type=float,
                        help="Learning rate decay")

    parser.add_argument("--char-embeddings", dest="char_embeddings", help="File from which to read in pretrained embeds")
    parser.add_argument("--char-embedding-dim", default=100, dest="char_embedding_dim", type=int,
                        help="Dimension of char embedding")

    parser.add_argument("--dataset", required=True, dest="dataset", help=".pkl file to use")
    parser.add_argument("--debug", dest="debug", default=False, action="store_true", help="Debug mode")

    parser.add_argument("--batch-size", default=20, dest="batch_size", type=int,
                        help="Minibatch size of training set")
                        
    parser.add_argument("--output-dir", default=os.path.join('output' , time.strftime("%Y-%m-%d-%H-%M-%S")), dest="output_dir",
                        help="Directory where to write logs / serialized models")
                        
    parser.add_argument("--save-checkpoint", dest="save_checkpoint", action="store_true", help="Save checkpoint during training")
    parser.add_argument("--old-model", dest="old_model", help="Path to old model for incremental training")
    
    parser.add_argument("--skip-dev", dest="skip_dev", action="store_true", help="Skip dev set, would save some time")
    
    parser.add_argument("--test", dest="test", action="store_true", help="Test mode")
    
    parser.add_argument("--use-bigram", dest="use_bigram", action="store_true", help="Use bigram feature")
    return parser.parse_args()