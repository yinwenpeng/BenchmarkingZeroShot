# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys
import codecs
import numpy as np
import torch
from collections import defaultdict
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss
from scipy.special import softmax
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

from pytorch_transformers.file_utils import PYTORCH_TRANSFORMERS_CACHE
from pytorch_transformers.modeling_bert import BertForSequenceClassification, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_transformers.tokenization_bert import BertTokenizer
from pytorch_transformers.optimization import AdamW

# from pytorch_transformers import *

from preprocess_yahoo import evaluate_Yahoo_zeroshot_SinglePhasePred
# import torch.optim as optimizer_wenpeng

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


type2hypothesis = {
0: ['it is related with society or culture', 'this text  describes something about an extended social group having a distinctive cultural and economic organization or a particular society at a particular time and place'],
1:['it is related with science or mathematics', 'this text  describes something about a particular branch of scientific knowledge or a science (or group of related sciences) dealing with the logic of quantity and shape and arrangement'],
2: ['it is related with health', 'this text  describes something about a healthy state of wellbeing free from disease'],
3: ['it is related with education or reference', 'this text  describes something about the activities of educating or instructing or activities that impart knowledge or skill or an indicator that orients you generally'],
4: ['it is related with computers or Internet', 'this text  describes something about a machine for performing calculations automatically or a computer network consisting of a worldwide network of computer networks that use the TCP/IP network protocols to facilitate data transmission and exchange'],
5: ['it is related with sports', 'this text  describes something about an active diversion requiring physical exertion and competition'],
6: ['it is related with business or finance', 'this text  describes something about a commercial or industrial enterprise and the people who constitute it or the commercial activity of providing funds and capital'],
7: ['it is related with entertainment or music', 'this text  describes something about an activity that is diverting and that holds the attention or an artistic form of auditory communication incorporating instrumental or vocal tones in a structured and continuous manner'],
8: ['it is related with family or relationships', 'this text  describes something about a social unit living together, primary social group; parents and children or a relation between people'],
9: ['it is related with politics or government', 'this text  describes something about social relations involving intrigue to gain authority or power or the organization that is the governing authority of a political unit']}

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""
    def get_train_examples_wenpeng(self, filename):
        readfile = codecs.open(filename, 'r', 'utf-8')
        line_co=0
        examples=[]
        for row in readfile:
            if line_co>0:
                line=row.strip().split('\t')
                guid = "train-"+line[0]
                text_a = line[1]
                text_b = line[2]
                label = line[-1]
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
                line_co+=1
            else:
                line_co+=1
                continue
        readfile.close()
        print('loaded training size:', line_co)
        return examples


    def get_examples_Yahoo_train(self, filename, size_limit_per_type):
        readfile = codecs.open(filename, 'r', 'utf-8')
        line_co=0
        exam_co = 0
        examples=[]
        label_list = []

        '''first get all the seen types, since we will only create pos and neg hypo in seen types'''
        seen_types = set()
        for row in readfile:
            line=row.strip().split('\t')
            if len(line)==2: # label_id, text
                type_index =  int(line[0])
                seen_types.add(type_index)
        readfile.close()

        readfile = codecs.open(filename, 'r', 'utf-8')
        type_load_size = defaultdict(int)
        for row in readfile:
            line=row.strip().split('\t')
            if len(line)==2: # label_id, text

                type_index =  int(line[0])
                if type_load_size.get(type_index,0)< size_limit_per_type:
                    for i in range(10):
                        hypo_list = type2hypothesis.get(i)
                        if i == type_index:
                            '''pos pair'''
                            for hypo in hypo_list:
                                guid = "train-"+str(exam_co)
                                text_a = line[1]
                                text_b = hypo
                                label = 'entailment' #if line[0] == '1' else 'not_entailment'
                                examples.append(
                                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
                                exam_co+=1
                        elif i in seen_types:
                            '''neg pair'''
                            for hypo in hypo_list:
                                guid = "train-"+str(exam_co)
                                text_a = line[1]
                                text_b = hypo
                                label = 'not_entailment' #if line[0] == '1' else 'not_entailment'
                                examples.append(
                                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
                                exam_co+=1
                    line_co+=1
                    if line_co % 10000 == 0:
                        print('loading training size:', line_co)

                    type_load_size[type_index]+=1
                else:
                    continue
        readfile.close()
        print('loaded size:', line_co)
        print('seen_types:', seen_types)
        return examples, seen_types


    def get_examples_Yahoo_test(self, filename, seen_types):
        readfile = codecs.open(filename, 'r', 'utf-8')
        line_co=0
        exam_co = 0
        examples=[]

        hypo_seen_str_indicator=[]
        hypo_2_type_index=[]
        for i in range(10):
            hypo_list = type2hypothesis.get(i)
            for hypo in hypo_list:
                hypo_2_type_index.append(i) # this hypo is for type i
                if i in seen_types:
                    hypo_seen_str_indicator.append('seen')# this hypo is for a seen type
                else:
                    hypo_seen_str_indicator.append('unseen')

        gold_label_list = []
        for row in readfile:
            line=row.strip().split('\t')
            if len(line)==2: # label_id, text

                type_index =  int(line[0])
                gold_label_list.append(type_index)
                for i in range(10):
                    hypo_list = type2hypothesis.get(i)
                    if i == type_index:
                        '''pos pair'''
                        for hypo in hypo_list:
                            guid = "test-"+str(exam_co)
                            text_a = line[1]
                            text_b = hypo
                            label = 'entailment' #if line[0] == '1' else 'not_entailment'
                            examples.append(
                                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
                            exam_co+=1
                    else:
                        '''neg pair'''
                        for hypo in hypo_list:
                            guid = "test-"+str(exam_co)
                            text_a = line[1]
                            text_b = hypo
                            label = 'not_entailment' #if line[0] == '1' else 'not_entailment'
                            examples.append(
                                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
                            exam_co+=1
                line_co+=1
                if line_co % 1000 == 0:
                    print('loading test size:', line_co)
                # if line_co == 1000:
                #     break


        readfile.close()
        print('loaded size:', line_co)
        return examples, gold_label_list, hypo_seen_str_indicator, hypo_2_type_index

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}

    premise_2_tokenzed={}
    hypothesis_2_tokenzed={}
    list_2_tokenizedID = {}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = premise_2_tokenzed.get(example.text_a)
        if tokens_a is None:
            tokens_a = tokenizer.tokenize(example.text_a)
            premise_2_tokenzed[example.text_a] = tokens_a

        tokens_b = premise_2_tokenzed.get(example.text_b)
        if tokens_b is None:
            tokens_b = tokenizer.tokenize(example.text_b)
            hypothesis_2_tokenzed[example.text_b] = tokens_b

        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

        tokens_A = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids_A = [0] * len(tokens_A)
        tokens_B = tokens_b + ["[SEP]"]
        segment_ids_B = [1] * (len(tokens_b) + 1)
        tokens = tokens_A+tokens_B
        segment_ids = segment_ids_A+segment_ids_B


        input_ids_A = list_2_tokenizedID.get(' '.join(tokens_A))
        if input_ids_A is None:
            input_ids_A = tokenizer.convert_tokens_to_ids(tokens_A)
            list_2_tokenizedID[' '.join(tokens_A)] = input_ids_A
        input_ids_B = list_2_tokenizedID.get(' '.join(tokens_B))
        if input_ids_B is None:
            input_ids_B = tokenizer.convert_tokens_to_ids(tokens_B)
            list_2_tokenizedID[' '.join(tokens_B)] = input_ids_B
        input_ids = input_ids_A + input_ids_B


        # tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        # segment_ids = [0] * len(tokens)
        #
        # tokens += tokens_b + ["[SEP]"]
        # segment_ids += [1] * (len(tokens_b) + 1)
        # input_ids = tokenizer.convert_tokens_to_ids(tokens)


        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features

# def convert_examples_to_features(examples, label_list, max_seq_length,
#                                  tokenizer, output_mode):
#     """Loads a data file into a list of `InputBatch`s."""
#
#     label_map = {label : i for i, label in enumerate(label_list)}
#
#     features = []
#     for (ex_index, example) in enumerate(examples):
#         if ex_index % 10000 == 0:
#             logger.info("Writing example %d of %d" % (ex_index, len(examples)))
#
#         tokens_a = tokenizer.tokenize(example.text_a)
#
#         tokens_b = None
#         if example.text_b:
#             tokens_b = tokenizer.tokenize(example.text_b)
#             # Modifies `tokens_a` and `tokens_b` in place so that the total
#             # length is less than the specified length.
#             # Account for [CLS], [SEP], [SEP] with "- 3"
#             _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
#         else:
#             # Account for [CLS] and [SEP] with "- 2"
#             if len(tokens_a) > max_seq_length - 2:
#                 tokens_a = tokens_a[:(max_seq_length - 2)]
#
#         tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
#         segment_ids = [0] * len(tokens)
#
#         if tokens_b:
#             tokens += tokens_b + ["[SEP]"]
#             segment_ids += [1] * (len(tokens_b) + 1)
#
#         input_ids = tokenizer.convert_tokens_to_ids(tokens)
#
#         # The mask has 1 for real tokens and 0 for padding tokens. Only real
#         # tokens are attended to.
#         input_mask = [1] * len(input_ids)
#
#         # Zero-pad up to the sequence length.
#         padding = [0] * (max_seq_length - len(input_ids))
#         input_ids += padding
#         input_mask += padding
#         segment_ids += padding
#
#         assert len(input_ids) == max_seq_length
#         assert len(input_mask) == max_seq_length
#         assert len(segment_ids) == max_seq_length
#
#         if output_mode == "classification":
#             label_id = label_map[example.label]
#         elif output_mode == "regression":
#             label_id = float(example.label)
#         else:
#             raise KeyError(output_mode)
#
#         if ex_index < 5:
#             logger.info("*** Example ***")
#             logger.info("guid: %s" % (example.guid))
#             logger.info("tokens: %s" % " ".join(
#                     [str(x) for x in tokens]))
#             logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
#             logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
#             logger.info(
#                     "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
#             logger.info("label: %s (id = %d)" % (example.label, label_id))
#
#         features.append(
#                 InputFeatures(input_ids=input_ids,
#                               input_mask=input_mask,
#                               segment_ids=segment_ids,
#                               label_id=label_id))
#     return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "sts-b":
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    elif task_name == "mnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mnli-mm":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "qnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "rte":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == 'F1':
        return {"f1": f1_score(y_true=labels, y_pred=preds)}
    else:
        raise KeyError(task_name)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=64,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=256,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()

    # if args.server_ip and args.server_port:
    #     # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
    #     import ptvsd
    #     print("Waiting for debugger attach")
    #     ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
    #     ptvsd.wait_for_attach()

    processors = {
        # "cola": ColaProcessor,
        # "mnli": MnliProcessor,
        # "mnli-mm": MnliMismatchedProcessor,
        # "mrpc": MrpcProcessor,
        # "sst-2": Sst2Processor,
        # "sts-b": StsbProcessor,
        # "qqp": QqpProcessor,
        # "qnli": QnliProcessor,
        "rte": RteProcessor
        # "wnli": WnliProcessor,
    }

    output_modes = {
        # "cola": "classification",
        # "mnli": "classification",
        # "mrpc": "classification",
        # "sst-2": "classification",
        # "sts-b": "regression",
        # "qqp": "classification",
        # "qnli": "classification",
        "rte": "classification"
        # "wnli": "classification",
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")


    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    output_mode = output_modes[task_name]

    label_list = processor.get_labels() #[0,1]
    num_labels = len(label_list)



    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        # train_examples = processor.get_train_examples_wenpeng('/home/wyin3/Datasets/glue_data/RTE/train.tsv')
        train_examples, seen_types = processor.get_examples_Yahoo_train('/export/home/Dataset/YahooClassification/yahoo_answers_csv/zero-shot-split/train_pu_half_v1.txt', 130000) #train_pu_half_v1.txt
        # seen_classes=[0,2,4,6,8]

        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_TRANSFORMERS_CACHE), 'distributed_{}'.format(args.local_rank))

    pretrain_model_dir = 'bert-base-uncased' #FineTuneOnCombined'# FineTuneOnMNLI
    model = BertForSequenceClassification.from_pretrained(pretrain_model_dir, num_labels=num_labels)
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_dir, do_lower_case=args.do_lower_case)

    if args.fp16:
        model.half()
    model.to(device)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        optimizer = AdamW(optimizer_grouped_parameters,
                             lr=args.learning_rate)
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    max_test_unseen_acc = 0.0
    max_dev_unseen_acc = 0.0
    max_dev_seen_acc = 0.0
    max_overall_acc = 0.0
    if args.do_train:
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer, output_mode)

        '''load dev set'''
        eval_examples, eval_label_list, eval_hypo_seen_str_indicator, eval_hypo_2_type_index = processor.get_examples_Yahoo_test('/export/home/Dataset/YahooClassification/yahoo_answers_csv/zero-shot-split/dev.txt', seen_types)
        eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer, output_mode)

        eval_all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        eval_all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        eval_all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        eval_all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

        eval_data = TensorDataset(eval_all_input_ids, eval_all_input_mask, eval_all_segment_ids, eval_all_label_ids)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        '''load test set'''
        test_examples, test_label_list, test_hypo_seen_str_indicator, test_hypo_2_type_index = processor.get_examples_Yahoo_test('/export/home/Dataset/YahooClassification/yahoo_answers_csv/zero-shot-split/test.txt', seen_types)
        test_features = convert_examples_to_features(
            test_examples, label_list, args.max_seq_length, tokenizer, output_mode)

        test_all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
        test_all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
        test_all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
        test_all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)

        test_data = TensorDataset(test_all_input_ids, test_all_input_mask, test_all_segment_ids, test_all_label_ids)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)


        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)

        if output_mode == "classification":
            all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        elif output_mode == "regression":
            all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.float)

        # print('train all_label_ids:', all_label_ids)
        # exit(0)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        train_sampler = RandomSampler(train_data)

        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        iter_co = 0
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                model.train()
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                logits = model(input_ids, segment_ids, input_mask, labels=None)
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits[0].view(-1, num_labels), label_ids.view(-1))

                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                iter_co+=1
                if iter_co %400==0:
                    '''
                    start evaluate on dev set after this epoch
                    '''
                    model.eval()

                    logger.info("***** Running evaluation *****")
                    logger.info("  Num examples = %d", len(eval_examples))
                    logger.info("  Batch size = %d", args.eval_batch_size)

                    eval_loss = 0
                    nb_eval_steps = 0
                    preds = []
                    print('Evaluating...')
                    for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
                        input_ids = input_ids.to(device)
                        input_mask = input_mask.to(device)
                        segment_ids = segment_ids.to(device)
                        label_ids = label_ids.to(device)

                        with torch.no_grad():
                            logits = model(input_ids, segment_ids, input_mask, labels=None)
                        logits = logits[0]

                        loss_fct = CrossEntropyLoss()
                        tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))

                        eval_loss += tmp_eval_loss.mean().item()
                        nb_eval_steps += 1
                        if len(preds) == 0:
                            preds.append(logits.detach().cpu().numpy())
                        else:
                            preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)

                    eval_loss = eval_loss / nb_eval_steps
                    preds = preds[0]

                    '''
                    preds: size*2 (entail, not_entail)
                    wenpeng added a softxmax so that each row is a prob vec
                    '''
                    pred_probs = softmax(preds,axis=1)[:,0]
                    pred_binary_labels_harsh = []
                    pred_binary_labels_loose = []
                    for i in range(preds.shape[0]):
                        if preds[i][0]>preds[i][1]+0.1:
                            pred_binary_labels_harsh.append(0)
                        else:
                            pred_binary_labels_harsh.append(1)
                        if preds[i][0]>preds[i][1]:
                            pred_binary_labels_loose.append(0)
                        else:
                            pred_binary_labels_loose.append(1)

                    seen_acc, unseen_acc = evaluate_Yahoo_zeroshot_SinglePhasePred(pred_probs, pred_binary_labels_harsh, pred_binary_labels_loose, eval_label_list, eval_hypo_seen_str_indicator, eval_hypo_2_type_index, seen_types)
                    # result = compute_metrics('F1', preds, all_label_ids.numpy())
                    loss = tr_loss/nb_tr_steps if args.do_train else None
                    # test_acc = mean_f1#result.get("f1")
                    if unseen_acc > max_dev_unseen_acc:
                        max_dev_unseen_acc = unseen_acc
                    print('\ndev seen_acc & acc_unseen:', seen_acc,unseen_acc, ' max_dev_unseen_acc:', max_dev_unseen_acc, '\n')
                    '''
                    start evaluate on test set after this epoch
                    '''
                    model.eval()

                    logger.info("***** Running testing *****")
                    logger.info("  Num examples = %d", len(test_examples))
                    logger.info("  Batch size = %d", args.eval_batch_size)

                    test_loss = 0
                    nb_test_steps = 0
                    preds = []
                    print('Testing...')
                    for input_ids, input_mask, segment_ids, label_ids in test_dataloader:
                        input_ids = input_ids.to(device)
                        input_mask = input_mask.to(device)
                        segment_ids = segment_ids.to(device)
                        label_ids = label_ids.to(device)

                        with torch.no_grad():
                            logits = model(input_ids, segment_ids, input_mask, labels=None)
                        logits = logits[0]
                        if len(preds) == 0:
                            preds.append(logits.detach().cpu().numpy())
                        else:
                            preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)

                    # eval_loss = eval_loss / nb_eval_steps
                    preds = preds[0]
                    pred_probs = softmax(preds,axis=1)[:,0]
                    pred_binary_labels_harsh = []
                    pred_binary_labels_loose = []
                    for i in range(preds.shape[0]):
                        if preds[i][0]>preds[i][1]+0.1:
                            pred_binary_labels_harsh.append(0)
                        else:
                            pred_binary_labels_harsh.append(1)
                        if preds[i][0]>preds[i][1]:
                            pred_binary_labels_loose.append(0)
                        else:
                            pred_binary_labels_loose.append(1)

                    seen_acc, unseen_acc = evaluate_Yahoo_zeroshot_SinglePhasePred(pred_probs, pred_binary_labels_harsh, pred_binary_labels_loose, test_label_list, test_hypo_seen_str_indicator, test_hypo_2_type_index, seen_types)
                    if unseen_acc > max_test_unseen_acc:
                        max_test_unseen_acc = unseen_acc
                    print('\n\n\t test seen_acc & acc_unseen:', seen_acc,unseen_acc, ' max_test_unseen_acc:', max_test_unseen_acc, '\n')

if __name__ == "__main__":
    main()
# CUDA_VISIBLE_DEVICES=1,2 python -u train_Yahoo_fine_tune_Bert_zeroshot.py --task_name rte --do_train --do_lower_case --bert_model bert-base-uncased --max_seq_length 128 --train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3 --data_dir '' --output_dir ''
