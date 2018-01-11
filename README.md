# smu
A simple statistical machine translation implementation of IBM models 1, 2 and 3 ([Brown et al., 1993](https://dl.acm.org/citation.cfm?id=972474)) supported by an n-gram language model.

## Requirements
Due to its lightweight implementation, the standard libraries of Python 3 are all you need.

## Data
Training of the IBM models requires a parallel corpus formatted in the following way:

```
Das ist aber ein schöner Beispielsatz . ||| What a nice example sentence .
Jeder Satz verdient eine eigene Zeile . ||| Every sentence deserves its own line .
```

Furthermore, the language model is trained using a monolingual corpus with each sentence on its own line. Input files for translation follow the same format. Nothing fancy.

## Models

There are three translation models which are implemented in this project:

* **IBM Model 1** simply learns word translation probabilities while treating all alignments equally. It can be found in `src/models.py` and trained using the `train_model1` method.
* **IBM Model 2** builds upon model 1 and learns translation probabilities as well as word alignemnts. It can be found in `src/models.py` and trained using the `train_model2` method.
* **IBM Model 3** is more complex still and learns translation, word alignment, fertility and null non-insertion probabilities . It can be found in `src/models.py` and trained using the `train_model3` method and is used by default when running the translation script.

Furthermore n-gram language models are also implemented and used during decoding with a backoff approach. By default, trigrams, bigrams and unigrams are learned and used in conjunction with IBM Model 3.

## Translation

To run a translation experiment simply `./src/run.sh` the bash script or execute the translation script directly:

```
$ python3 translate.py ../data/train.de-en ../data/train.en ../data/input.de output_prefix
```
During the translation process, files containing the probability distributions' values are created in the working directory with the output prefix. Finally, an output file `output_prefix_output.txt` is created.

Help is available by running the above script without arguments.
