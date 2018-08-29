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
Instance_digit = collections.namedtuple("Instance_digit", ["sentence_array", "tag_ids"])

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
    # Make up some training data
    training_data = [(
        "充 满 活 力 的 热 门 音 乐 ，".split(),
        "B E B E S B E B E S".split()
    ), (
        "铺 着 石 板 的 广 场 中 搭 起 舞 台 ，".split(),
        "S S B E S B E S B E B E S".split()
    )]
    training_data_digit = []
    tag_to_ix = {'S': 0, 'B': 1, 'E': 2, 'M': 3}
    char_to_ix = {}
    for sentence, tags in training_data:
        sentence_array = []
        for char in sentence:
            if char not in char_to_ix:
                sentence_array.append(len(char_to_ix))
                char_to_ix[char] = len(char_to_ix)
            else:
                sentence_array.append(char_to_ix[char])
        tag_ids = [tag_to_ix[t] for t in tags]
        training_data_digit.append(Instance_digit(sentence_array, tag_ids))
    return training_data_digit, training_data_digit, training_data_digit, char_to_ix, tag_to_ix

    # dataset = pickle.load(open(args.dataset, "rb"))

    # # Get mappings
    # t2i = dataset["t2i"]
    # c2i = dataset["c2i"]
    # # Inverse mappings
    # i2t = utils.to_id_list(t2i)
    # i2c = utils.to_id_list(c2i)
    # # Get datasets
    # training_instances = dataset["training_instances"]
    # dev_instances = dataset["dev_instances"]
    # test_instances = dataset["test_instances"]

    # if args.debug:
    #     training_instances = training_instances[:1600]
    #     dev_instances = dev_instances[:200]
    #     test_instances = test_instances[:200]

    # return training_instances, dev_instances, test_instances, c2i, t2i


def complete_collection(element_to_ix, element_list):
    element_id_list = []
    for element in element_list:
        if element not in element_to_ix:
            element_to_ix[element] = len(element_to_ix)
        element_id_list.append(element_to_ix[element])
    return element_to_ix, element_id_list


def init_model(char_to_ix, tag_to_ix, START_CHAR_ID, STOP_CHAR_ID, START_TAG_ID, STOP_TAG_ID):
    if args.old_model is not None:
        model = torch.load(args.old_model)

    else:
        if args.char_embeddings is not None:
            char_embeddings = utils.read_pretrained_embeddings(args.char_embeddings, char_to_ix)
            EMBEDDING_DIM = char_embeddings.shape[1]
        else:
            char_embeddings = None
            EMBEDDING_DIM = args.char_embeddings_dim
        model = BiLSTM_CRF(len(char_to_ix), len(tag_to_ix), START_CHAR_ID, STOP_CHAR_ID, START_TAG_ID, STOP_TAG_ID,
                            args.use_bigram, args.hidden_dim, args.dropout, EMBEDDING_DIM, char_embeddings)

    return processor.to_cuda_if_available(model)


def train(model, training_data, optimizer):
    model.train()

    num_batches = math.ceil(len(training_data) / args.batch_size)
    bar = utils.Progbar(target=num_batches)
    train_loss = 0.0
    train_total_instances = 0

    for batch_id, batch in enumerate(utils.minibatches(training_data, args.batch_size)):
        model.zero_grad()
        
        for sentence, tags in batch:
            sentence_in = processor.tensor(sentence)
            targets = processor.tensor(tags)

            loss = model.neg_log_likelihood(sentence_in, targets)
            loss.backward()

            train_loss += loss
            train_total_instances += 1

        optimizer.step()

        bar.update(batch_id + 1, exact=[("train loss", train_loss / train_total_instances)])

    if args.save_checkpoint:
        save_model(model)


def evaluate(model, eval_data, dataset_name):
    model.eval()
    
    num_batches = math.ceil(len(eval_data) / args.batch_size)
    bar = utils.Progbar(target=num_batches)
    eval_score = 0.0
    eval_total_instances = 0
    eval_total_characters = 0
    eval_correct_characters = 0

    with torch.no_grad():
        for batch_id, batch in enumerate(utils.minibatches(eval_data, args.batch_size)):
            for sentence, tags in batch:
                score, tag_out = model(processor.tensor(sentence))
                if len(tag_out) != len(tags):
                    raise IndexError('Size of output tag sequence differs from that of reference.')
                length = len(tags)
                correct = [tag_out[i] == tags[i] for i in range(1, length - 1)].count(1)
                eval_score += score
                eval_total_instances += 1
                eval_total_characters += length
                eval_correct_characters += correct
            bar.update(batch_id + 1, exact=[("eval score", eval_score / eval_total_instances)])

        logger.info('{} dataset accuracy: {}'.format(dataset_name, eval_correct_characters/eval_total_characters))


def save_model(model):
    filename = os.path.join(args.output_dir, 'model.pt')
    utils.ensure_folder(filename)
    torch.save(model, filename)


args = arguments.parse_args()
logger = init_logger()
logger.info(args)


def main():
    training_data, dev_data, test_data, char_to_ix, tag_to_ix = load_datasets()

    START_TAG = "<START>"
    STOP_TAG = "<STOP>"
    char_to_ix, [START_CHAR_ID, STOP_CHAR_ID] = complete_collection(char_to_ix, [START_TAG, STOP_TAG])
    tag_to_ix, [START_TAG_ID, STOP_TAG_ID] = complete_collection(tag_to_ix, [START_TAG, STOP_TAG])

    model = init_model(char_to_ix, tag_to_ix, START_CHAR_ID, STOP_CHAR_ID, START_TAG_ID, STOP_TAG_ID)

    if not args.test:
        # Train the model
        for epoch in range(args.num_epochs):
            learning_rate = args.learning_rate / (1 + epoch)
            optimizer = optim.SGD([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
            logger.info('Epoch: {}/{}. Learning rate:{}'.format(epoch+1, args.num_epochs, learning_rate))
            train(model, training_data, optimizer)
            if not args.skip_dev:
                evaluate(model, dev_data, 'Dev')

    # Check predictions after training
    evaluate(model, test_data, 'Test')
    # We got it!
    

if __name__ == '__main__':
    main()