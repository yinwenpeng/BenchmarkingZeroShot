# BenchmarkingZeroShot

Hi, this repository contains the code and the data for the EMNLP2019 paper "Benchmarking Zero-shot Text Classification: Datasets, Evaluation and Entailment Approach"

To download the dataset for "topic detection", "emotion detection" and "situation detection", pls go to this URL: https://drive.google.com/open?id=1qGmyEVD19ruvLLz9J0QGV7rsZPFEz2Az

To download the wikipedia data and three pretrained entailment models (RTE, MNLI, FEVER), pls go to this URL:
https://drive.google.com/file/d/1ILCQR_y-OSTdgkz45LP7JsHcelEsvoIn/view?usp=sharing

Any questions can be sent to mr.yinwenpeng@gmail.com

If you play this benchmark dataset, please cite:

    @inproceedings{yinroth2019zeroshot,
        title={Benchmarking Zero-shot Text Classification: Datasets, Evaluation and Entailment Approach},
        author={Wenpeng Yin, Jamaal Hay and Dan Roth},
        booktitle={{EMNLP}},
        url = {https://arxiv.org/abs/1909.00161},
        year={2019}
    }

Requirements:

    Pytorch
    Transformer (pytorch): https://github.com/huggingface/transformers
    GPU

Commandline to rerun the code (take "baseline_wiki_based_emotion.py" as an example):

    CUDA_VISIBLE_DEVICES=1 python -u baseline_wiki_based_emotion.py --task_name rte --do_train --do_lower_case --bert_model bert-base-uncased --max_seq_length 128 --train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3 --data_dir '' --output_dir ''

Very importance step before running:
Since our code was written in "pytorch-transformer" -- the old verion of Huggingface Transformer, pls
1) update the "pytorch-transformer" into "transformer" before running the code. For example:

Now it is:

    from pytorch_transformers.file_utils import PYTORCH_TRANSFORMERS_CACHE
    from pytorch_transformers.modeling_bert import BertForSequenceClassification, BertConfig, WEIGHTS_NAME, CONFIG_NAME
    from pytorch_transformers.tokenization_bert import BertTokenizer
    from pytorch_transformers.optimization import AdamW

change  to be:

    from transformers.file_utils import PYTORCH_TRANSFORMERS_CACHE
    from transformers.modeling_bert import BertForSequenceClassification
    from transformers.tokenization_bert import BertTokenizer
    from transformers.optimization import AdamW

2) the new Transformer's function "BertForSequenceClassification" has parameter order slightly different with the prior "pytorch_transformer". The current version is:

        def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                    position_ids=None, head_mask=None, inputs_embeds=None, labels=None):

 the old version is:

       def forward(self, input_ids=None, token_type_ids=None, attention_mask=None,
                  position_ids=None, head_mask=None, inputs_embeds=None, labels=None):

 namely the token_ids and mask are exchanged. So, you need to change the input order (only for "token_type_ids" and "attention_mask") when ever you call the model. For example, my code currently is:

             logits = model(input_ids, input_mask,segment_ids, labels=None)

 change it to be:

             logits = model(input_ids, segment_ids, input_mask, labels=None)
             
