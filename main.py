import logging
import os
import sys
import time
import collections
import pickle

import torch
import torch.nn as nn
import torch.optim as optim

import arguments
import utils
from model import BiLSTM_CRF
from model import tensor

Instance = collections.namedtuple("Instance", ["sentence", "tags"])
# Instance_digit = collections.namedtuple("Instance_digit", ["sentence_array", "tag_ids"])


args = arguments.parse_args()


def load_datasets():
    dataset = pickle.load(open(args.dataset, "rb"))

    # Get mappings
    t2i = dataset["t2i"]
    c2i = dataset["c2i"]
    # Inverse mappings
    i2t = utils.to_id_list(t2i)
    i2c = utils.to_id_list(c2i)
    # Get datasets
    training_instances = dataset["training_instances"]
    dev_instances = dataset["dev_instances"]
    test_instances = dataset["test_instances"]

    if args.debug:
        training_instances = training_instances[:800]
        dev_instances = dev_instances[:100]
        test_instances = test_instances[:100]

    return training_instances, dev_instances, test_instances, c2i, t2i
    # Make up some training data
    # training_data = [(
    #     "充 满 活 力 的 热 门 音 乐 ，".split(),
    #     "B E B E S B E B E S".split()
    # ), (
    #     "铺 着 石 板 的 广 场 中 搭 起 舞 台 ，".split(),
    #     "S S B E S B E S B E B E S".split()
    # )]
    # training_data_digit = []

    # tag_to_ix = {'S': 0, 'B': 1, 'E': 2, 'M': 3}

    # char_to_ix = {}
    # for sentence, tags in training_data:
    #     sentence_array = []
    #     for char in sentence:
    #         if char not in char_to_ix:
    #             sentence_array.append(len(char_to_ix))
    #             char_to_ix[char] = len(char_to_ix)
    #         else:
    #             sentence_array.append(char_to_ix[char])
    #     tag_ids = [tag_to_ix[t] for t in tags]
    #     training_data_digit.append(Instance_digit(sentence_array, tag_ids))

    # return training_data_digit, training_data_digit, training_data_digit, char_to_ix, tag_to_ix


def complete_tags(tag_to_ix):
    START_TAG = "<START>"
    STOP_TAG = "<STOP>"
    if START_TAG not in tag_to_ix:
        tag_to_ix[START_TAG] = len(tag_to_ix)
    if STOP_TAG not in tag_to_ix:
        tag_to_ix[STOP_TAG] = len(tag_to_ix)
    return tag_to_ix, tag_to_ix[START_TAG], tag_to_ix[STOP_TAG]


def init_model(char_to_ix, tag_to_ix, START_TAG_ID, STOP_TAG_ID):
    if args.char_embeddings is not None:
        char_embeddings = utils.read_pretrained_embeddings(args.char_embeddings, char_to_ix)
        EMBEDDING_DIM = char_embeddings.shape[1]
        model = BiLSTM_CRF(len(char_to_ix), len(tag_to_ix), START_TAG_ID, STOP_TAG_ID,
                            args.hidden_dim, args.dropout, EMBEDDING_DIM, char_embeddings)
    else:
        EMBEDDING_DIM = args.char_embedding_dim
        model = BiLSTM_CRF(len(char_to_ix), len(tag_to_ix), START_TAG_ID, STOP_TAG_ID,
                            args.hidden_dim, args.dropout, EMBEDDING_DIM)
    
    if torch.cuda.is_available():
        model.cuda()
    
    return model


def train(model, optimizer, batch_size, training_data):
    for batch_id, batch in enumerate(utils.minibatches(training_data, batch_size)):
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each batch
        model.zero_grad()

        for sentence, tags in batch:
            # Step 2. Get our inputs ready for the network, that is,
            # turn them into Tensors of char indices.
            sentence_in = tensor(sentence)
            targets = tensor(tags)

            # Step 3. Run our forward pass.
            loss = model.neg_log_likelihood(sentence_in, targets)

            # Step 4. Compute the loss, gradients
            loss.backward()
        
        # Step 5. Update the parameters by calling optimizer.step()
        optimizer.step()


def save_model(model):
    filename = os.path.join(args.output_dir, 'model.pt')
    utils.ensure_folder(filename)
    torch.save(model, filename)


def evaluate(model, test_data, dataset):
    model.eval()
    with torch.no_grad():
        total = 0
        correct = 0
        for sentence, tags in test_data:
            score, tag_seq = model(tensor(sentence))
            if len(tag_seq) != len(tags):
                raise IndexError('Size of output tag sequence differs from that of reference.')
            totl = len(tags)
            crct = [tag_seq[i] == tags[i] for i in range(1, totl - 1)].count(1)
            total += totl
            correct += crct
        print('{} dataset accuracy: {}'.format(dataset, correct/total))


def main():
    training_data, dev_data, test_data, char_to_ix, tag_to_ix = load_datasets()
    tag_to_ix, START_TAG_ID, STOP_TAG_ID = complete_tags(tag_to_ix)

    if args.old_model is not None:
        model = torch.load(args.old_model)
    else:
        model = init_model(char_to_ix, tag_to_ix, START_TAG_ID, STOP_TAG_ID)

    # Check predictions before training
    evaluate(model, test_data, 'Test')

    # Make sure prepare_sequence from earlier in the LSTM section is loaded
    for epoch in range(
            args.num_epochs):

        learning_rate = args.learning_rate / (1 + epoch)
        print('Epoch: {}/{}. Learning rate:{}'.format(epoch, args.num_epochs, learning_rate))

        optimizer = optim.SGD([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
        train(model, optimizer, args.batch_size, training_data)
        evaluate(model, dev_data, 'Dev')

    # Check predictions after training
    evaluate(model, test_data, 'Test')
    # We got it!

    if args.save_model:
        save_model(model)
    

if __name__ == '__main__':
    main()