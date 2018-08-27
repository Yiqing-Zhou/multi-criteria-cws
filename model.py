import torch
import torch.nn as nn
import processor

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tagset_size, start_char_id, stop_char_id, start_tag_id, stop_tag_id,
                    use_bigram, hidden_dim, dropout, embedding_dim, char_embedding=None):
        super(BiLSTM_CRF, self).__init__()
        self.vocab_size = vocab_size
        self.tagset_size = tagset_size

        self.start_char_id = start_char_id
        self.stop_char_id = stop_char_id
        self.start_tag_id = start_tag_id
        self.stop_tag_id = stop_tag_id

        self.use_bigram = use_bigram
        self.hidden_dim = hidden_dim

        # Train or load pretrained char_embedding
        if char_embedding is None:
            self.char_embeds = nn.Embedding(vocab_size, embedding_dim)
        else:
            char_embedding = processor.tensor(char_embedding, dtype=torch.float)
            self.char_embeds = nn.Embedding.from_pretrained(char_embedding)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            dropout=dropout, num_layers=1, bidirectional=True)

        if use_bigram:
            hidden_dim += embedding_dim * 2
        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Sequential(
            nn.Linear(hidden_dim, tagset_size),
            nn.Tanh(),
            nn.Linear(tagset_size, tagset_size)
        )

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            processor.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[start_tag_id, :] = -10000
        self.transitions.data[:, stop_tag_id] = -10000

    def init_hidden(self):
        return (processor.randn(2, 1, self.hidden_dim // 2),
                processor.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = processor.full((1, self.tagset_size), -10000.)
        # self.start_tag has all of the score.
        init_alphas[0][self.start_tag_id] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward processor.tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.stop_tag_id]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.char_embeds(sentence)
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1), self.hidden)
        lstm_out = lstm_out.squeeze()
        if self.use_bigram:
            embeds_shift_left = torch.cat(
                [self.char_embeds(processor.tensor([self.start_char_id])),
                embeds[:-1]])
            embeds_shift_right = torch.cat(
                [embeds[1:],
                self.char_embeds(processor.tensor([self.stop_char_id])))
            embeds_bi1 = (embeds_shift_left + embeds) / 2
            embeds_bi2 = (embeds_shift_right + embeds) / 2
            lstm_out = torch.cat([embeds_bi1, lstm_out, embeds_bi2], 1)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = processor.zeros(1)
        tags = torch.cat([processor.tensor([self.start_tag_id], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.stop_tag_id, tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = processor.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.start_tag_id] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to self.stop_tag
        terminal_var = forward_var + self.transitions[self.stop_tag_id]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.start_tag_id  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq