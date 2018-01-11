'''
    Decoders
'''
from utils import *
from math import log, exp
from sys import stdout

#
# functions
#

def decode_lexical(corpus, distribution, verbose=False) :
    if verbose : print(" - decoding lexical - ")
    res = []
    dist_t = distribution
    for index_sen, sentence_f in enumerate(corpus) :
        if (verbose) and ((index_sen+1)%5 == 0):
            stdout.write(('\rdecoding : %d of %d sentences'+(' '*10)) % (index_sen+1, len(corpus.sentences)))
            stdout.flush()
        sentence_e = []
        # lexical translation step
        for index_f, token_f in enumerate(sentence_f) :
            sorted_options = dist_t.get_options_sorted((token_f,))
            if len(sorted_options) > 0 :
                sentence_e.append(sorted_options[0][0])
            else :
                sentence_e.append(token_f)
        res.append(sentence_e)
    if verbose : print("\n - decoding complete - ")
    return res

def decode_lexical_lm(corpus, dist_t, dists_lm, lm_weights, verbose=False) :
    if verbose : print(" - decoding lexical (with language model) - ")
    res = []
    n_length = len(dists_lm.keys())
    for index_sen, sentence_f in enumerate(corpus) :
        if (verbose) and ((index_sen+1)%5 == 0):
            stdout.write(('\rdecoding : %d of %d sentences'+(' '*10)) % (index_sen+1, len(corpus)))
            stdout.flush()
        sentence_f = ["<s>"] + sentence_f + ["<\s>"] # add start and end tags
        sentence_e = []
        for index_f in range(len(sentence_f)) :
            ngram_options = sentence_e[max(index_f-n_length+1,0):index_f] # get the n-1 previous tokens
            ngram_options.append(dist_t.get_options_sorted((sentence_f[index_f],))[:10]) # add top 10 hyps as n-th token
            max_option = sentence_f[index_f] # initialize as foreign token
            max_prob = None
            if len(ngram_options[-1]) > 0 :
                max_option = ngram_options[-1][0][0] # top lexical translation if it exists
            for token_n in ngram_options[-1] : # iterate over all hyps
                ngram_prob = 0.
                for n in range(1,len(ngram_options)+1) :
                    cur_ngram = (token_n[0],)
                    if n > 1 :
                        cur_ngram += tuple(ngram_options[-n:-1])
                    # cur_prob = dists_lm[n].get_probability(cur_ngram)
                    # if cur_prob is None :
                    #     cur_prob = 1/len(dists_lm[n].lookup.items())
                    if cur_ngram in dists_lm[n] :
                        cur_prob = dists_lm[n][cur_ngram] # get ngram-probability if it exists
                    else :
                        cur_prob = 1/len(dists_lm[n].keys()) # 1/count_unique(n) else
                    ngram_prob += cur_prob * lm_weights[n] # apply weights
                ngram_prob += token_n[1] # add to lexical probability
                if (max_prob is None) or (ngram_prob > max_prob) :
                    max_prob = ngram_prob
                    max_option = token_n[0]
            sentence_e.append(max_option) # append maximum option
        sentence_e = sentence_e[1:-1] # remove start and end tags
        res.append(sentence_e)
    if verbose : print("\n - decoding complete - ")
    return res

def decode_model3_lm(corpus, dist_t, dist_d, dist_f, prob_p0, dists_lm, lm_weights, verbose=False) :
    print(" - decoding with model 3 distributions (+ language model) - ")
    res = []
    for index_sen, sentence_f in enumerate(corpus) :
        if (verbose) :
            stdout.write(('\rdecoding : %d of %d sentences'+(' '*10)) % (index_sen+1, len(corpus)))
            stdout.flush()
        sentence_e = []
        sentence_hyp = [] # initialize empty hypothesis
        # fertility and lexical translation step
        for index_f, token_f in enumerate(sentence_f) :
            fert_options = dist_f.get_options_sorted((token_f,))[:1] # get top fertility option
            lex_options = dist_t.get_options_sorted((token_f,))[:5] # get top 5 lexical translation options
            if len(fert_options) < 1 :
                fert_options = [(1,0.)]
            if len(lex_options) < 1 :
                lex_options = [(token_f,0.)]
            if fert_options[0][0] == 0 :
                sentence_hyp.append([('',0.)])
            else :
                if (token_f not in ['.', '!', '?', ',', "'", '-']) and (len(lex_options) > 1) :
                    lex_options = [lex_option for lex_option in lex_options if lex_option[0] not in ['.', '!', '?', ',', "'", '-']]
                sentence_hyp += [lex_options for i in range(fert_options[0][0])]
        # distortion step
        for index_hyp in range(1, len(sentence_hyp)+1) :
            token_hyp = sentence_hyp[index_hyp-1]
            length_f = len(sentence_f)+1
            length_hyp = len(sentence_hyp)+1
            sentence_dist = [token_hyp for token_hyp in sentence_hyp]
            align_options = dist_d.get_options_sorted((index_hyp, length_hyp, length_f))[:1] # best option
            if len(align_options) > 0 :
                align_index = align_options[0][0]-1
                sentence_dist[align_options[0][0]-1] = token_hyp
        sentence_hyp = sentence_dist
        # apply language model
        n_length = len(dists_lm.keys())
        sentence_lm = [[('<s>',0.)]] + sentence_hyp + [[('</s>',0.)]] # add start and end tags
        for index_hyp, token_hyp in enumerate(sentence_lm) :
            max_hyp = token_hyp[0]
            max_prob = None
            for hyp in token_hyp : # iterate over all hyps
                hyp_prob = 0.
                for n in range(1,min(n_length, index_hyp+1)+1) :
                    cur_ngram = (hyp[0],)
                    if n > 1 :
                        cur_ngram += tuple(sentence_e[-n:-1])
                    if cur_ngram in dists_lm[n] :
                        hyp_prob = exp(dists_lm[n][cur_ngram]) * lm_weights[n] # get ngram-probability if it exists
                    else :
                        hyp_prob = 1/len(dists_lm[n].keys()) * lm_weights[n] # 1/count_unique(n) else
                hyp_prob += exp(hyp[1]) # add to lexical probability
                if (max_prob is None) or (hyp_prob > max_prob) :
                    max_prob = hyp_prob
                    max_hyp = hyp
            if max_hyp[0] != '' :
                sentence_e.append(max_hyp[0]) # append maximum option
        sentence_e = sentence_e[1:len(sentence_e)-1] # remove start and end tags
        res.append(sentence_e)
    if verbose : print("\n - decoding complete - ")
    return res

def decode_brute(corpus, dist_t, dist_d, dist_f, prob_p0, dists_lm, lm_weights, verbose=False) :
    if verbose : print(" - decoding as brute with model 3 distributions (+ language model) - ")
    res = []
    for index_sen, sentence_f in enumerate(corpus) :
        if (verbose) and ((index_sen+1)%5 == 0):
            stdout.write(('\rdecoding : %d of %d sentences'+(' '*10)) % (index_sen+1, len(corpus)))
            stdout.flush()
        sentence_e = []
        sentence_hyps = [([],0.)] # initialize empty hypothesis
        for index_f, token_f in enumerate(sentence_f) :
            # fertility and lexical translation step
            fert_options = dist_f.get_options_sorted((token_f,))[:3] # get top 3 fertility options
            lex_options = dist_t.get_options_sorted((token_f,))[:5] # get top 5 lexical translation options
            if len(fert_options) < 1 :
                fert_options = [(1,0.)]
            if len(lex_options) < 1 :
                lex_options = [(token_f,0.)]
            updated_sentence_hyps = []
            for sentence_hyp in sentence_hyps :
                for fert_option in fert_options :
                    for lex_option in lex_options :
                        updated_sentence = sentence_hyp[0] + [lex_option[0] for i in range(fert_option[0])] # append translated token with fertility frequency
                        updated_prob = sentence_hyp[1] + fert_option[1] + lex_option[1] # log probability
                        updated_sentence_hyps.append((updated_sentence, updated_prob))
            sentence_hyps = updated_sentence_hyps
        # distortion step
        updated_sentence_hyps = []
        for sentence_hyp in sentence_hyps :
            length_f = len(sentence_f)
            length_hyp = len(sentence_hyp[0])
            distorted_sentence_hyps = [sentence_hyp]
            for index_hyp, token_hyp in enumerate(sentence_hyp[0]) :
                align_options = dist_d.get_options_sorted((index_hyp, length_hyp, length_f)) # best option
                updated_distorted_sentence_hyps = []
                for distorted_sentence_hyp in distorted_sentence_hyps :
                    for align_option in align_options :
                        distorted_sentence = distorted_sentence_hyp[0]
                        distorted_sentence[align_option[0]] = token_hyp
                        distorted_prob = distorted_sentence_hyp[1] + align_option[1]
                        updated_distorted_sentence_hyps.append((distorted_sentence, distorted_prob))
                distorted_sentence_hyps = updated_distorted_sentence_hyps
            updated_sentence_hyps.append(distorted_sentence_hyp)
        sentence_hyps = updated_sentence_hyps
        # apply language model
        n_length = len(dists_lm.keys())
        max_index = None
        for index_hyp in range(len(sentence_hyps)) :
            sentence_lm = ['<s>'] + sentence_hyps[index_hyp][0] + ['</s>'] # add start and end tags
            sentence_prob = sentence_hyps[index_hyp][1]
            # for index_hyp, token_hyp in enumerate(sentence_lm) :
            #     for n in range(1, min(n_length, index_hyp+1)) :
            #         cur_ngram = (token_hyp,)
            #         if n > 1:
            #             cur_ngram += tuple(sentence_lm[-n:index_hyp])
            #         if cur_ngram in dists_lm[n] :
            #             sentence_prob += dists_lm[n][cur_ngram] * lm_weights[n] # get ngram-probability if it exists
            #         else :
            #             sentence_prob += log(1/len(dists_lm[n].keys())) * lm_weights[n] # 1/count_unique(n) else
            sentence_hyps[index_hyp] = (sentence_hyps[index_hyp][0], sentence_prob)
            if (max_index is None) or (sentence_prob > sentence_hyps[max_index][1]) :
                max_index = index_hyp
        sentence_e = sentence_hyps[max_index][0]
        print(sentence_f)
        print(sentence_e)
        res.append(sentence_e)
    if verbose : print("\n - decoding complete - ")
    return res
