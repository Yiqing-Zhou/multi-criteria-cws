import dynet as dy
import numpy as np

class BiLSTM_CRF:
    def __init__(self, vocab_size, tagset_size, start_char_id, stop_char_id, start_tag_id, stop_tag_id,
                    use_bigram, hidden_dim, dropout, embedding_dim, char_embeddings=None):
        self.model = dy.Model()
        self.vocab_size = vocab_size
        self.tagset_size = tagset_size

        self.start_char_id = start_char_id
        self.stop_char_id = stop_char_id
        self.start_tag_id = start_tag_id
        self.stop_tag_id = stop_tag_id

        self.use_bigram = use_bigram
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.ce_update = True

        # Train or load pretrained char_embeddings
        self.char_lookup = self.model.add_lookup_parameters((vocab_size, embedding_dim))
        if char_embeddings is not None:
            self.char_lookup.init_from_array(char_embeddings)
        self.bi_lstm = dy.BiRNNBuilder(1, embedding_dim, hidden_dim, self.model, dy.LSTMBuilder)
        
        # Matrix that maps from Bi-LSTM output to num tags
        if use_bigram:
            self.lstm_to_tags_params = self.model.add_parameters((tagset_size, hidden_dim + embedding_dim * 2))
        else:
            self.lstm_to_tags_params = self.model.add_parameters((tagset_size, hidden_dim))
        self.lstm_to_tags_bias = self.model.add_parameters(tagset_size)
        self.mlp_out = self.model.add_parameters((tagset_size, tagset_size))
        self.mlp_out_bias = self.model.add_parameters(tagset_size)

        # Transition matrix for tagging layer, [i,j] is score of transitioning to i from j
        self.transitions = self.model.add_lookup_parameters((tagset_size, tagset_size))

    def enable_dropout(self):
        self.bi_lstm.set_dropout(self.dropout)

    def disable_dropout(self):
        self.bi_lstm.disable_dropout()

    def build_tagging_graph(self, sentence):
        dy.renew_cg()

        embeds = [dy.lookup(self.char_lookup, c, update=self.ce_update) for c in sentence]

        lstm_out = self.bi_lstm.transduce(embeds)

        H = dy.parameter(self.lstm_to_tags_params)
        Hb = dy.parameter(self.lstm_to_tags_bias)
        O = dy.parameter(self.mlp_out)
        Ob = dy.parameter(self.mlp_out_bias)
        scores = []
        if self.use_bigram:
            for rep, char in zip(lstm_out, sentence):
                embeds_shift_left = dy.concatenate([
                    dy.lookup(self5.char_lookup, self.start_char_id, update=self.ce_update),
                    embeds[:-1]
                ])
                embeds_shift_right = dy.concatenate([
                    embeds[1:],
                    dy.lookup(self.char_lookup, self.stop_char_id, update=self.ce_update)
                ])
                embeds_bi1 = (embeds + embeds_shift_left)/2
                embeds_bi2 = (embeds + embeds_shift_right)/2
                if self.dropout is not None:
                    embeds_bi1 = dy.dropout(embeds_bi1, self.dropout)
                    embeds_bi2 = dy.dropout(embeds_bi2, self.dropout)
                score_t = O * dy.tanh(H * dy.concatenate(
                    [embeds_bi1,
                     embeds,
                     embeds_bi2]) + Hb) + Ob
                scores.append(score_t)
        else:
            for rep in lstm_out:
                score_t = O * dy.tanh(H * rep + Hb) + Ob
                scores.append(score_t)

        return scores

    def score_sentence(self, observations, tags):
        assert len(observations) == len(tags)
        score_seq = [0]
        score = dy.scalarInput(0)
        tags = [self.start_tag_id] + tags
        for i, obs in enumerate(observations):
            score = score + dy.pick(self.transitions[tags[i + 1]], tags[i]) + dy.pick(obs, tags[i + 1])
            score_seq.append(score.value())
        score = score + dy.pick(self.transitions[self.stop_tag_id], tags[-1])
        return score

    def viterbi_loss(self, sentence, gold_tags, use_margins=True):
        observations = self.build_tagging_graph(sentence)
        viterbi_tags, viterbi_score = self.viterbi_decoding(observations, gold_tags, use_margins)
        if viterbi_tags != gold_tags:
            gold_score = self.score_sentence(observations, gold_tags)
            return (viterbi_score - gold_score), viterbi_tags
        else:
            return dy.scalarInput(0), viterbi_tags

    def neg_log_loss(self, sentence, tags):
        observations = self.build_tagging_graph(sentence)
        gold_score = self.score_sentence(observations, tags)
        forward_score = self.forward(observations)
        return forward_score - gold_score

    def forward(self, observations):

        def log_sum_exp(scores):
            npval = scores.npvalue()
            argmax_score = np.argmax(npval)
            max_score_expr = dy.pick(scores, argmax_score)
            max_score_expr_broadcast = dy.concatenate([max_score_expr] * self.tagset_size)
            return max_score_expr + dy.log(dy.sum_dim(dy.transpose(dy.exp(scores - max_score_expr_broadcast)), [1]))

        init_alphas = [-1e10] * self.tagset_size
        init_alphas[self.start_tag_id] = 0
        for_expr = dy.inputVector(init_alphas)
        for obs in observations:
            alphas_t = []
            for next_tag in range(self.tagset_size):
                obs_broadcast = dy.concatenate([dy.pick(obs, next_tag)] * self.tagset_size)
                next_tag_expr = for_expr + self.transitions[next_tag] + obs_broadcast
                alphas_t.append(log_sum_exp(next_tag_expr))
            for_expr = dy.concatenate(alphas_t)
        terminal_expr = for_expr + self.transitions[self.stop_tag_id]
        alpha = log_sum_exp(terminal_expr)
        return alpha

    def viterbi_decoding(self, observations, gold_tags, use_margins):
        backpointers = []
        init_vvars = [-1e10] * self.tagset_size
        init_vvars[self.start_tag_id] = 0  # <Start> has all the probability
        for_expr = dy.inputVector(init_vvars)
        trans_exprs = [self.transitions[idx] for idx in range(self.tagset_size)]
        for gold, obs in zip(gold_tags, observations):
            bptrs_t = []
            vvars_t = []
            for next_tag in range(self.tagset_size):
                next_tag_expr = for_expr + trans_exprs[next_tag]
                next_tag_arr = next_tag_expr.npvalue()
                best_tag_id = np.argmax(next_tag_arr)
                bptrs_t.append(best_tag_id)
                vvars_t.append(dy.pick(next_tag_expr, best_tag_id))
            for_expr = dy.concatenate(vvars_t) + obs

            # optional margin adaptation
            if use_margins and self.margins != 0:
                adjust = [self.margins] * self.tagset_size
                adjust[gold] = 0
                for_expr = for_expr + dy.inputVector(adjust)
            backpointers.append(bptrs_t)
        # Perform final transition to terminal
        terminal_expr = for_expr + trans_exprs[self.stop_tag_id]
        terminal_arr = terminal_expr.npvalue()
        best_tag_id = np.argmax(terminal_arr)
        path_score = dy.pick(terminal_expr, best_tag_id)
        # Reverse over the backpointers to get the best path
        best_path = [best_tag_id]  # Start with the tag that was best for terminal
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop()  # Remove the start symbol
        best_path.reverse()
        assert start == self.start_tag_id
        # Return best path and best path's score
        return best_path, path_score
