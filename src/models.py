'''
    SMT Tools
'''
from utils import *
from math import log, exp
from collections import defaultdict
from sys import stdout

#
# functions
#

def train_model1(corpus, iterations, verbose=False) :
    '''
        EM training function according to IBM Model 1

        returns the translation probability t = {(e,f) : prob}
    '''
    if verbose : print(" - training IBM Model 1 - ")
    # initialize t uniformly
    t = defaultdict(lambda: 1./corpus.count_unique_f())
    # training loop
    for i in range(iterations) :
        count = defaultdict(lambda:0.)
        total = defaultdict(lambda:0.)
        stotal = {}
        for index_pair, pair in enumerate(corpus) :
            if (verbose) and ( ((index_pair+1)%100 == 0) or (i+1 == iterations) ):
                stdout.write(('\rtraining iteration : %d of %d | %d of %d sentence pairs | %d token pairs'+(' '*10)) % (i+1, iterations, index_pair+1, len(corpus), len(t.keys())))
                stdout.flush()
            # insert null token
            sentence_f = [""] + pair[0]
            sentence_e = [""] + pair[1]
            # compute normalization
            for token_e in sentence_e :
                stotal[token_e] = 0
                for token_f in sentence_f :
                    stotal[token_e] += t[(token_e,token_f)]
            # collect counts
            for token_e in sentence_e :
                for token_f in sentence_f :
                    count[(token_e,token_f)] += t[(token_e,token_f)] / stotal[token_e]
                    total[token_f] += t[(token_e,token_f)] / stotal[token_e]
                    if total[token_f] == 0 :
                        print(token_f, total[token_f])
        # probability estimation
        for token_e, token_f in corpus.get_token_pairs() :
            t[(token_e,token_f)] = count[(token_e,token_f)] / total[token_f]
        corpus.reset_iter()
    if verbose : print("\n - training of IBM Model 1 complete - ")
    return dict(t)

def train_model2(corpus, iterations, verbose=False) :
    '''
        EM training function according to IBM Model 2

        returns (t, a)
            the translation probability t = {(e,f) : prob}
            the alignment probability a = {(i,j,l_e,l_f) : prob }
    '''
    if verbose : print(" - training IBM Model 2 - ")
    t = {}
    a = {}
    # initialize t according to Model 1
    if verbose : print("initialize t according to Model 1...")
    t = train_model1(corpus, iterations, verbose=verbose)
    # initialize a uniformly
    for pair in corpus :
        length_f = len(pair[0])+1
        length_e = len(pair[1])+1
        for index_f in range(length_f) :
            for index_e in range(length_e) :
                a[(index_f,index_e,length_e,length_f)] = 1./(length_f+1)
    # training loop
    for i in range(iterations) :
        count_t = defaultdict(lambda:0)
        total_t = defaultdict(lambda:0)
        count_a = defaultdict(lambda:0)
        total_a = defaultdict(lambda:0)
        stotal = {}
        corpus.reset_iter()
        for index_pair, pair in enumerate(corpus) :
            if (verbose) and ( ((index_pair+1)%100 == 0) or (i+1 == iterations) ):
                stdout.write(('\rtraining iteration : %d of %d | %d of %d sentence pairs | %d alignments'+(' '*10)) % (i+1, iterations, index_pair+1, len(corpus), len(a.keys())))
                stdout.flush()
            sentence_f = [""] + pair[0] # insert null token
            sentence_e = [""] + pair[1]
            length_f = len(sentence_f)
            length_e = len(sentence_e)
            # compute normalization
            for index_e, token_e in enumerate(sentence_e) :
                stotal[token_e] = 0
                for index_f, token_f in enumerate(sentence_f) :
                    stotal[token_e] += t[(token_e,token_f)] * a[(index_f,index_e,length_e,length_f)]
            # collect counts
            for index_e, token_e in enumerate(sentence_e) :
                for index_f, token_f in enumerate(sentence_f) :
                    update_c = t[(token_e,token_f)] * a[(index_f,index_e,length_e,length_f)]/stotal[token_e]
                    count_t[(token_e,token_f)] += update_c
                    total_t[token_f] += update_c
                    count_a[(index_f,index_e,length_e,length_f)] += update_c
                    total_a[(index_e,length_e,length_f)] += update_c
        # probability estimation
        for token_e, token_f in t.keys() :
            t[(token_e, token_f)] = count_t[(token_e, token_f)] / total_t[token_f]
        for alignment in a.keys() :
            a[alignment] = count_a[alignment] / total_a[alignment[1:]]
    if verbose : print("\n - training of IBM Model 2 complete - ")
    return dict(t), dict(a)

def train_model3(corpus, iterations, verbose=False) :
    '''
        EM training function according to IBM Model 3

        returns (t, d, f, n)
            the translation probability t = {(e,f) : prob}
            the distortion probability d = {(j,i,l_e,l_f) : prob }
            the fertility probability f = {(n,f) : prob }
            the null non-insertion probability p0 = prob
    '''
    if verbose : print(" - training IBM Model 3 - ")
    t = {}
    d = {}
    f = {}
    p0 = None
    # initialize t,d according to Model 2
    if verbose : print("initialize t, d according to Model 2...")
    t, d = train_model2(corpus, iterations*2, verbose=verbose)
    # remap distributions t, d
    for pair in t :
         # convert and filter 0 probabilites
        if t[pair] > 0 : t[pair] = log(t[pair])
    remap_d = {}
    for align in d :
        # convert and filter 0 probabilites
        if d[align] > 0 : remap_d[(align[1], align[0], align[2], align[3])] = log(d[align])
    d = remap_d
    # training loop
    for i in range(iterations) :
        count_t = defaultdict(lambda:0)
        total_t = defaultdict(lambda:0)
        count_d = defaultdict(lambda:0)
        total_d = defaultdict(lambda:0)
        count_f = defaultdict(lambda:0)
        total_f = defaultdict(lambda:0)
        count_null = 0
        count_p1 = 0
        count_p0 = 0
        stotal = {}
        corpus.reset_iter()
        for index_pair, pair in enumerate(corpus) :
            if (verbose) :
                stdout.write(('\rtraining iteration : %d of %d | %d of %d sentence pairs | %d alignments | %d fertiliy values |'+(' '*10)) % (i+1, iterations, index_pair+1, len(corpus), len(d.keys()), len(f.keys())))
                stdout.flush()
            # initialize local pair variables
            sentence_f = [""] + pair[0] # insert null token
            sentence_e = [""] + pair[1]
            length_f = len(sentence_f)
            length_e = len(sentence_e)
            # get sample alignments
            sample_alignments = sample_model3(sentence_e, sentence_f, t, d)
            if sample_alignments is None :
                # skip if no valid alignments are found
                continue
            sample_probs = []
            count_total = 0
            valid_alignments = []
            for align in sample_alignments :
                align_prob = align.get_probability(d)
                for index_f, token_f in enumerate(sentence_f) :
                    token_e = sentence_e[align.get_index_e(index_f)]
                    if (token_e, token_f) in t :
                        cur_sample_prob = t[(token_e, token_f)]+align_prob # log probability
                        valid_alignments.append(align)
                        sample_probs.append(cur_sample_prob)
            sample_alignments = valid_alignments
            min_sample_prob = min(sample_probs)
            for index_prob in range(len(sample_probs)) :
                sample_probs[index_prob] = -1*min_sample_prob + sample_probs[index_prob]
            count_norm = -1*min_sample_prob
            for index_align, align in enumerate(sample_alignments) :
                # normalize log probabilities as count
                if sample_probs[index_align] == 0 :
                    count = 1
                else :
                    count = sample_probs[index_align] / count_norm
                for index_f, token_f in enumerate(sentence_f) :
                    index_e = align.get_index_e(index_f)
                    token_e = sentence_e[index_e]
                    count_t[(token_e, token_f)] += count
                    total_t[token_f] += count
                    count_d[(index_e, index_f, length_e, length_f)] += count
                    total_d[(index_f, length_e, length_f)] += count
                    if index_e == 0 :
                        count_null += 1
                count_p1 += count_null * count
                count_p0 += (length_e - 2 * count_null) * count
                for index_f in range(length_f) :
                    fertility = 0
                    for index_e in range(length_e) :
                        if (index_e == align.get_index_e(index_f)) and (align.get_index_e(index_f) != 0) :
                            fertility += 1
                    count_f[(fertility, sentence_f[index_f])] += count
                    total_f[sentence_f[index_f]] += count
        # probability estimation
        t = {}
        d = {}
        f = {}
        for token_e, token_f in count_t.keys() :
            cur_prob_t = count_t[(token_e, token_f)] / total_t[token_f]
            if cur_prob_t > 0 : t[(token_e, token_f)] = log(cur_prob_t) # log probability
        for index_e, index_f, length_e, length_f in count_d.keys() :
            cur_prob_d = count_d[(index_e, index_f, length_e, length_f)] / total_d[(index_f, length_e, length_f)]
            if cur_prob_d > 0 : d[(index_e, index_f, length_e, length_f)] = log(cur_prob_d) # log probability
        for fertility, token_f in count_f.keys() :
            cur_prob_f = count_f[(fertility, token_f)] / total_f[token_f]
            if cur_prob_f > 0 : f[(fertility, token_f)] = log(cur_prob_f) # log probability
        p1 = count_p1 / (count_p0 + count_p1)
        p0 = 1 - p1
    if verbose : print("\n - training of IBM Model 3 complete - ")
    return dict(t), dict(d), dict(f), p0

def sample_model3(sentence_e, sentence_f, prob_t, prob_d) :
    res = []
    length_e = len(sentence_e)
    length_f = len(sentence_f)
    # determine argmax over index_e
    argmax_token_alignments = []
    for index_f in range(length_f) :
        max_alignment = (None, None)
        for try_e in range(length_e) :
            cur_prob_t = None
            if (sentence_e[try_e], sentence_f[index_f]) in prob_t.keys() :
                cur_prob_t = prob_t[(sentence_e[try_e], sentence_f[index_f])]
            cur_prob_d = None
            if (try_e, index_f, length_e, length_f) in prob_d.keys() :
                cur_prob_d = prob_d[(try_e, index_f, length_e, length_f)]
            if (cur_prob_t is not None) and (cur_prob_d is not None) :
                cur_prob = cur_prob_t + cur_prob_d # log probability
                if (max_alignment[1] is None) or (cur_prob > max_alignment[1]):
                    max_alignment = (try_e, cur_prob)
        if max_alignment[0] is None:
            argmax_token_alignments = None
            break
        argmax_token_alignments.append(max_alignment[0])
    if argmax_token_alignments is not None :
        cur_alignment = alignment(length_e, length_f, argmax_token_alignments)
        res.append(cur_alignment)
    else :
        # cur_alignment = alignment(length_e, length_f)
        return None
    # perform sampling
    # for index_pegged in range(length_f) :
    #     # cur_alignment = cur_alignment.hillclimb(prob_d, index_pegged)
    #     # if cur_alignment not in res :
    #     #     res.append(cur_alignment)
    #     for neighbor in cur_alignment.get_neighbors(index_pegged) :
    #         if (neighbor not in res) and (neighbor.get_probability(prob_d) is not None) :
    #             res.append(neighbor)
    return res

def train_lm(corpus, n_length, verbose=False) :
    if verbose : print(" - training "+str(n_length)+"-gram language model - ")
    res = {}
    # collect counts
    counts = {}
    for n in range(1,n_length+1) :
        res[n] = {}
        counts[n] = {}
    for index_sen, sentence in enumerate(corpus) :
        if (verbose) and ((index_sen+1)%100 == 0):
            stdout.write(('\rtraining : %d of %d sentences'+(' '*10)) % (index_sen+1, len(corpus)))
            stdout.flush()
        sentence = ["<s>"] + sentence + ["</s>"]
        for index_token in range(len(sentence)) :
            for n in range(1, n_length+1):
                ngram = tuple(sentence[index_token:(index_token+n)])
                if index_token+n <= len(sentence) :
                    if ngram in counts[n] :
                        counts[n][ngram] += 1
                    else :
                        counts[n][ngram] = 1
    # probability estimation
    if verbose : print("\nestimating probabilites...")
    for n in range(1,n_length+1) :
        for ngram in counts[n] :
            if n > 1 :
                res[n][(ngram[len(ngram)-1],)+ngram[:-1]] = log(counts[n][ngram] / counts[n-1][ngram[:n-1]])
            else :
                res[n][ngram] = log(counts[n][ngram] / len(counts[n].keys()))
    if verbose : print(" - training complete - ")
    return res
