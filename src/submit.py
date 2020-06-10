# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import os
import random
import sys

import numpy as np
import torch

from torch.utils.data import (DataLoader, SequentialSampler, TensorDataset)
from tqdm import tqdm

from models.modeling_bert import QuestionAnswering, Config
from utils.tokenization import BertTokenizer
from utils.korquad_utils import read_squad_examples, convert_examples_to_features, write_predictions, RawResult

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CHK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "korquad_3.bin")
CONFIG_PATH = os.path.join(MODEL_PATH, "bert_small.json")
VOCAB_PATH = os.path.join(MODEL_PATH, "ko_vocab_32k.txt")

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def main(input, output):
    max_seq_length = 512
    doc_stride = 64
    max_query_length = 64
    batch_size = 16
    n_best_size = 20
    max_answer_length = 30
    seed = 42
    fp16 = False

    device = torch.device("cpu")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    tokenizer = BertTokenizer(vocab_file=VOCAB_PATH, max_len=max_seq_length, do_basic_tokenize=True)
    eval_examples = read_squad_examples(input_file=input, is_training=False, version_2_with_negative=False)
    eval_features = convert_examples_to_features(
        examples=eval_examples,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        doc_stride=doc_stride,
        max_query_length=max_query_length,
        is_training=False)

    # Prepare model
    config = Config.from_json_file(CONFIG_PATH)
    model = QuestionAnswering(config)
    if fp16 is True:
      model.half()
    model.load_state_dict(torch.load(CHK_PATH, map_location=device))
    model.to(device)
    logger.info("***** Running training *****")
    logger.info("  Num orig examples = %d", len(eval_examples))
    logger.info("  Num split examples = %d", len(eval_features))
    logger.info("  Batch size = %d", batch_size)

    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)

    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)

    model.eval()
    all_results = []
    logger.info("Start evaluating")
    for input_ids, input_mask, segment_ids, example_indices in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        with torch.no_grad():
            batch_start_logits, batch_end_logits = model(input_ids, segment_ids, input_mask)
        for i, example_index in enumerate(example_indices):
            start_logits = batch_start_logits[i].detach().cpu().tolist()
            end_logits = batch_end_logits[i].detach().cpu().tolist()
            eval_feature = eval_features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            all_results.append(RawResult(unique_id=unique_id,
                                         start_logits=start_logits,
                                         end_logits=end_logits))
    output_nbest_file = os.path.join("nbest_predictions.json")
    write_predictions(eval_examples, eval_features, all_results,
                        n_best_size, max_answer_length,
                        False, output, output_nbest_file,
                        None, False, False, 0.0)

if __name__ == "__main__":
    print(sys.argv)
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    data_name = input_file.split('.')[0]
    print('input file:', input_file)
    print('output file:', output_file)
    main(input_file, output_file)
