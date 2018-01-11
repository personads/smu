#!/bin/usr/python3
import argparse
from utils import *
from models import *
from decoders import *

# argument parsing
arg_parser = argparse.ArgumentParser(description='performs training and decoding steps for the smt system')
arg_parser.add_argument('corpus_parallel', help='path to the parallel corpus')
arg_parser.add_argument('corpus_lm', help='path to the language model corpus')
arg_parser.add_argument('corpus_foreign', help='path to the foreign corpus')
arg_parser.add_argument('out_prefix', help='output prefix')
args = arg_parser.parse_args()

print(" - translating from scratch - ")
print()
print("importing parallel corpus...")
corpus_ef = corpus_parallel(args.corpus_parallel)
prob_model3 = train_model3(corpus_ef, 2, verbose=True)
corpus_ef = None
print("loading probability distributions...")
dist_t = distribution(prob_model3[0])
dist_d = distribution(prob_model3[1])
dist_f = distribution(prob_model3[2])
prob_p0 = prob_model3[3]
print("pruning probability distributions...")
dist_t.prune_probability(-10)
dist_d.prune_probability(-10)
dist_f.prune_probability(-10)
print("exporting probabilities (for safety)...")
export_probabilities(dist_t.get_probabilities(),args.out_prefix+"_t_export.txt")
export_probabilities(dist_d.get_probabilities(),args.out_prefix+"_d_export.txt")
export_probabilities(dist_f.get_probabilities(),args.out_prefix+"_f_export.txt")
export_probabilities({"p0":prob_p0},args.out_prefix+"_p0_export.txt")
print()

print("importing language model corpus...")
corpus_lm = corpus(args.corpus_lm)
prob_lm = train_lm(corpus_lm, 3, verbose=True)
corpus_lm = None
print("exporting probabilities (for safety)...")
export_probabilities(prob_lm[1],args.out_prefix+"_lm1_export.txt")
export_probabilities(prob_lm[2],args.out_prefix+"_lm2_export.txt")
export_probabilities(prob_lm[3],args.out_prefix+"_lm3_export.txt")
print()

print("importing foreign corpus...")
corpus_f = corpus(args.corpus_foreign)
res = decode_model3_lm(corpus_f, dist_t, dist_d, dist_f, prob_p0, prob_lm, {1:0, 2:0.6, 3:0.4}, verbose=True)
print("exporting output...")
export_sentences(res, args.out_prefix+"_output.txt")
print()
print(" - translation complete - ")
