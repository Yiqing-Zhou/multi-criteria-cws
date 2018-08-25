import logging
import os
import math
import collections
import pickle

import torch
import torch.nn as nn
import torch.optim as optim

import arguments
import utils
import processor
from model import BiLSTM_CRF

Instance = collections.namedtuple("Instance", ["sentence", "tags"])


def init_logger():
    logger = logging.getLogger()

    log_formatter = logging.Formatter("%(message)s")

    filename = os.path.join(args.output_dir, 'info.log')
    utils.ensure_folder(filename)
    file_handler = logging.FileHandler(filename, mode='w')
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    return logger


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


def complete_tags(tag_to_ix):
    START_TAG = "<START>"
    STOP_TAG = "<STOP>"
    if START_TAG not in tag_to_ix:
        tag_to_ix[START_TAG] = len(tag_to_ix)
    if STOP_TAG not in tag_to_ix:
        tag_to_ix[STOP_TAG] = len(tag_to_ix)
    return tag_to_ix, tag_to_ix[START_TAG], tag_to_ix[STOP_TAG]


def init_model(char_to_ix, tag_to_ix, START_TAG_ID, STOP_TAG_ID):
    if args.old_model is not None:
        model = torch.load(args.old_model)

    else:
        if args.char_embeddings is not None:
            char_embeddings = utils.read_pretrained_embeddings(args.char_embeddings, char_to_ix)
            EMBEDDING_DIM = char_embeddings.shape[1]
        else:
            char_embeddings = None
            EMBEDDING_DIM = args.char_embedding_dim
        model = BiLSTM_CRF(len(char_to_ix), len(tag_to_ix), START_TAG_ID, STOP_TAG_ID,
                            args.hidden_dim, args.dropout, EMBEDDING_DIM, char_embeddings)

    return processor.to_cuda_if_available(model)


def train(model, training_data, learning_rate):
    model.train()

    num_batches = math.ceil(len(training_data) / args.batch_size)
    bar = utils.Progbar(target=num_batches)

    optimizer = optim.SGD([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
    for batch_id, batch in enumerate(utils.minibatches(training_data, args.batch_size)):
        model.zero_grad()
        
        for sentence, tags in batch:
            sentence_in = processor.tensor(sentence)
            targets = processor.tensor(tags)

            loss = model.neg_log_likelihood(sentence_in, targets)
            loss.backward()
        optimizer.step()

        bar.update(batch_id + 1, exact=[("train loss", loss.item())])

    if args.save_checkpoint:
        save_model(model)


def evaluate(model, test_data, dataset_name):
    model.eval()
    with torch.no_grad():
        total = 0
        correct = 0
        for sentence, tags in test_data:
            score, tag_seq = model(processor.tensor(sentence))
            if len(tag_seq) != len(tags):
                raise IndexError('Size of output tag sequence differs from that of reference.')
            totl = len(tags)
            crct = [tag_seq[i] == tags[i] for i in range(1, totl - 1)].count(1)
            total += totl
            correct += crct
        logger.info('{} dataset accuracy: {}'.format(dataset_name, correct/total))


def save_model(model):
    filename = os.path.join(args.output_dir, 'model.pt')
    utils.ensure_folder(filename)
    torch.save(model, filename)

args = arguments.parse_args()
logger = init_logger()


def main():
    training_data, dev_data, test_data, char_to_ix, tag_to_ix = load_datasets()
    tag_to_ix, START_TAG_ID, STOP_TAG_ID = complete_tags(tag_to_ix)

    model = init_model(char_to_ix, tag_to_ix, START_TAG_ID, STOP_TAG_ID)

    # Check predictions before training
    evaluate(model, test_data, 'Test')

    # Train the model
    for epoch in range(
            args.num_epochs):
        learning_rate = args.learning_rate / (1 + epoch)
        logger.info('Epoch: {}/{}. Learning rate:{}'.format(epoch, args.num_epochs, learning_rate))
        train(model, training_data, learning_rate)
        if not args.skip_dev:
            evaluate(model, dev_data, 'Dev')

    # Check predictions after training
    evaluate(model, test_data, 'Test')
    # We got it!
    

if __name__ == '__main__':
    main()