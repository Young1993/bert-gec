from subword.apply_bpe import *
from bert_nmt import interactive_api as corrector
from scripts.detok import  detok
import argparse

def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="learn BPE-based word segmentation")

    parser.add_argument(
        '--codes', '-c',
        help="File with BPE codes (created by learn_bpe.py).")
    parser.add_argument(
        '--name', default='./gec-pseudodata/bpe/bpe_code.trg.dict_bpe8000')
    parser.add_argument(
        '--merges', '-m', type=int, default=-1,
        metavar='INT',
        help="Use this many BPE operations (<= number of learned symbols)"+
             "default: Apply all the learned merge operations")
    parser.add_argument(
        '--separator', '-s', type=str, default='@@', metavar='STR',
        help="Separator between non-final subword units (default: '%(default)s'))")
    parser.add_argument(
        '--vocabulary', type=argparse.FileType('r'), default=None,
        metavar="PATH",
        help="Vocabulary file (built with get_vocab.py). If provided, this script reverts any merge operations that produce an OOV.")
    parser.add_argument(
        '--vocabulary-threshold', type=int, default=None,
        metavar="INT",
        help="Vocabulary threshold. If vocabulary is provided, any word with frequency < threshold will be treated as OOV")
    parser.add_argument(
        '--glossaries', type=str, nargs='+', default=None,
        metavar="STR",
        help="Glossaries. Words matching any of the words/regex provided in glossaries will not be affected "+
             "by the BPE (i.e. they will neither be broken into subwords, nor concatenated with other subwords. "+
             "Can be provided as a list of words/regex after the --glossaries argument. Enclose each regex in quotes.")
    parser.add_argument(
        '--seed', type=int, default=None,
        metavar="S",
        help="Random seed for the random number generators (e.g. for BPE dropout with --dropout).")
    parser.add_argument(
        '--num-workers', type=int, default=1,
        help="Number of processors to process texts, only supported in Python3. If -1, set `multiprocessing.cpu_count()`. (default: %(default)s)")

    return parser

parser = create_parser()
args = parser.parse_args()

# bpe
args.codes = codecs.open(args.name, encoding='utf-8')
line = 'In most parts of the world, the volume of traffic is growing at an alarming rate. In the form of an assignment, discuss about the main traffic problems in your country, their causes and possible solutions.'
bpe = BPE(args.codes, args.merges, args.separator, None, args.glossaries)
bpe_bert = bpe.process_line(line)
print(bpe_bert)

# detok
test_bert = detok(line)
print(test_bert)

my_args = corrector.Myargs()
correction_sub = corrector.Correction(my_args)

