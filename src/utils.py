'''
    Utilities for the SMT System
'''
#
# classes
#

class corpus :
    '''
        Corpus helper class
    '''
    def __init__(self, path) :
        self.path = path
        self.line_count = 0
        self.lines_counted = False
        self.fop = None
        self.file_iter = None
        self.iter_line = None
        try :
            self.fop = open(path, 'r', encoding='utf8')
            self.file_iter = iter(self.fop)
            try :
                self.iter_line = next(self.file_iter)
                self.iter_line = self.iter_line.strip().split()
                self.line_count += 1
            except StopIteration :
                self.iter_line = None
        except IOError :
            if self.fop :
                self.fop.close()
            print('Error : IOError occured while opening "' + path + '"')

    def __iter__(self) :
        return self

    def __next__(self) :
        res = self.iter_line
        try :
            self.iter_line = next(self.file_iter)
            self.iter_line = self.iter_line.strip().split()
            if not self.lines_counted :
                self.line_count += 1
        except StopIteration :
            self.iter_line = None
            self.lines_counted = True
        if res is None :
            raise StopIteration
        return res

    def __len__(self) :
        return self.line_count

    def reset_iter(self) :
        try :
            self.fop.close()
            self.fop = open(self.path, 'r', encoding='utf8')
            self.file_iter = iter(self.fop)
            try :
                self.iter_line = next(self.file_iter)
                self.iter_line = self.iter_line.strip().split()
            except StopIteration :
                self.iter_line = None
        except IOError :
            if self.fop :
                self.fop.close()
            print('Error : IOError occured while opening "' + self.path + '"')

class corpus_parallel :
    '''
        Parallel corpus helper class
    '''
    def __init__(self, path) :
        self.path = path
        self.pair_count = 0
        self.pairs_counted = False
        self.fop = None
        self.file_iter = None
        self.iter_pair = None
        try :
            self.fop = open(path, 'r', encoding='utf8')
            self.file_iter = iter(self.fop)
            try :
                raw_line = next(self.file_iter)
                self.iter_pair = [ sentence.strip().split() for sentence in raw_line.split("|||") ]
                self.pair_count += 1
            except StopIteration as sierr :
                self.iter_pair = None
        except IOError :
            if self.fop :
                self.fop.close()
            print('Error : IOError occured while opening "' + path + '"')
        self._unique_token_pairs = None
        self._count_unique_e = None
        self._count_unique_f = None

    def __iter__(self) :
        return self

    def __next__(self) :
        res = self.iter_pair
        if res is None :
            raise StopIteration
        try :
            raw_line = next(self.file_iter)
            self.iter_pair = [ sentence.strip().split() for sentence in raw_line.split("|||") ]
            if not self.pairs_counted :
                self.pair_count += 1
        except StopIteration :
            self.iter_pair = None
            self.pairs_counted = True
        return res

    def __len__(self) :
        return self.pair_count

    def get_token_pairs(self) :
        res = None
        if self._unique_token_pairs is None :
            self._get_unique()
        res = self._unique_token_pairs
        return res

    def count_unique_f(self) :
        res = None
        if self._count_unique_f is None:
            self._get_unique()
        res = self._count_unique_f
        return res

    def count_unique_e(self) :
        res = None
        if self._count_unique_e is None:
            self._get_unique()
        res = self._count_unique_e
        return res

    def _get_unique(self) :
        unique_token_pairs = set()
        unique_tokens_e = set()
        unique_tokens_f = set()
        try :
            with open(self.path, 'r', encoding='utf8') as fop :
                for raw_line in fop :
                    pair = [ sentence.strip().split() for sentence in raw_line.split("|||") ]
                    for token_e in pair[1] :
                        unique_tokens_e.add(token_e)
                        for token_f in pair[0] :
                            unique_tokens_f.add(token_f)
                            unique_token_pairs.add((token_e, token_f))
        except IOError :
            print('Error : IOError occured while opening "' + self.path + '"')
        self._unique_token_pairs = unique_token_pairs
        self._count_unique_e = len(unique_tokens_e)
        self._count_unique_f = len(unique_tokens_f)

    def reset_iter(self) :
        try :
            self.fop.close()
            self.fop = open(self.path, 'r', encoding='utf8')
            self.file_iter = iter(self.fop)
            try :
                raw_line = next(self.file_iter)
                self.iter_pair = [ sentence.strip().split() for sentence in raw_line.split("|||") ]
            except StopIteration :
                self.iter_pair = None
        except IOError :
            if self.fop :
                self.fop.close()
            print('Error : IOError occured while opening "' + self.path + '"')

class distribution :
    '''
        Probability distribution helper class
    '''
    def __init__(self, probabilities) :
        # save probabilities in a lookup dict for much faster recall
        # p(arg|given) => lookup[given] = [(arg, prob), ...]
        self.lookup = {}
        for probability in probabilities.keys() :
            if probability[1:] in self.lookup :
                self.lookup[probability[1:]].append((probability[0], probabilities[probability]))
            else :
                self.lookup[probability[1:]] = [(probability[0], probabilities[probability])]

    def get_probabilities(self) :
        '''
            returns a dict of all probabilities p[(arg|given)] = prob
        '''
        res = {}
        for key in self.lookup.keys() :
            for option in self.lookup[key] :
                res[(option[0],)+key] = option[1]
        return res

    def get_probability(self, query) :
        res = 0.
        options = self.get_options(query[1:])
        if query[0] in options.keys() :
            res = options[query[0]]
        return res

    def get_options(self, query) :
        '''
            returns a dict of options given a query p(res|query)
            arguments
                query : tuple
        '''
        res = {}
        if query in self.lookup :
            for option in self.lookup[query] :
                res[(option[0])] = option[1]
        return res

    def get_options_sorted(self, query) :
        '''
            returns a list of options given a query p(res|query) sorted by p
        '''
        res = self.get_options(query)
        return sorted(res.items(), key=lambda item: item[1], reverse=True)

    def prune_probability(self, threshold) :
        '''
            prunes options according to a probability threshold value
        '''
        for key in self.lookup.keys() :
            for index_option, option in enumerate(self.lookup[key]) :
                if option[1] <= threshold :
                    del(self.lookup[key][index_option])

class alignment :
    '''
        Alignment helper class
    '''
    def __init__(self, length_e, length_f, token_alignments=None) :
        self.length_e = length_e
        self.length_f = length_f
        if token_alignments :
            self.token_alignments = token_alignments
        else :
            # default is a n:n mapping up to length_e
            self.token_alignments = [min(i, length_e-1) for i in range(1, length_f+1)]

    def __eq__(self, other) :
        res = False
        res = (self.token_alignments == other.token_alignments) and (self.length_e == other.length_e)
        return res

    def get_index_e(self, index_f) :
        return self.token_alignments[index_f]

    def set_index_e(self, index_f, index_e) :
        self.token_alignments[index_f] = index_e

    def get_probability(self, prob_d) :
        res = 0.
        for index_f, index_e in enumerate(self.token_alignments) :
            if (index_e, index_f, self.length_e, self.length_f) in prob_d :
                res += prob_d[(index_e, index_f, self.length_e, self.length_f)] # log probability
            else :
                res = None
                break
        return res

    def get_neighbors(self, pegged) :
        res = []
        token_alignments = []
        for index_f in range(self.length_f) :
            if index_f != pegged :
                for index_e in range(self.length_e) :
                    move_list = list(self.token_alignments)
                    move_list[index_f] = index_e
                    if move_list not in res :
                        token_alignments.append(move_list)
        for index_f in range(self.length_f) :
            for index_swap in range(self.length_f) :
                if (index_f != index_swap) and (index_f != pegged) and (index_swap != pegged) :
                    swap_list = list(self.token_alignments)
                    swap_list[index_swap] = self.token_alignments[index_f]
                    swap_list[index_f] = self.token_alignments[index_swap]
                    if swap_list not in res :
                        token_alignments.append(swap_list)
        for token_alignment in token_alignments :
            res.append(alignment(self.length_e, self.length_f, token_alignment))
        return res

    def hillclimb(self, prob_d, pegged) :
        res = self
        res_prob = res.get_probability(prob_d)
        for neighbor in self.get_neighbors(pegged) :
            neighbor_prob = neighbor.get_probability(prob_d)
            if neighbor_prob is None :
                continue
            elif (res_prob is None) or (neighbor_prob > res_prob) :
                res = neighbor
                res_prob = neighbor_prob
        return res

#
# functions
#

def export_probabilities(dist, path) :
    '''
        exports a distribution to text file
    '''
    with open(path, "w", encoding="utf8") as fop :
        for key in dist.keys() :
            fop_line = ""
            for key_index in key :
                fop_line += str(key_index) + " ||| "
            fop_line += str(dist[key]) + "\n"
            fop.write(fop_line)

def import_probabilities(path, datatype=str()) :
    '''
        imports a distribution from ||| separated text file
        arguments
            path : path to distribution file
            datatype=str() : the type the entries should be cast to
    '''
    res = {}
    with open(path, "r", encoding="utf8") as fop :
        for line in fop :
            fop_items = [ item.strip() for item in line.split("|||") ]
            for fop_index, fop_item in enumerate(fop_items) :
                try :
                    fop_items[fop_index] = type(datatype)(fop_item) # cast fop_item to datatype
                except (TypeError, ValueError) :
                    fop_items[fop_index] = fop_item # if not possible, just use str
            res[tuple(fop_items[:-1])] = float(fop_items[len(fop_items)-1])
    return res

def export_sentences(sentences, path) :
    '''
        exports tokenized sentences to path
    '''
    with open(path, "w", encoding="utf8") as fop:
        for sentence in sentences :
            fop_line = ""
            for token in sentence :
                fop_line += token + " "
            fop_line = fop_line[:-1] + "\n"
            fop.write(fop_line)
