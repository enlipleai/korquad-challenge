<p align="center"><img src="./img/logos.png"></p>

# 한국어 AI 언어모델 튜닝대회
본 repository는 한국어 AI 언어모델 튜닝대회에서 사용하는 Small Size의 사전학습 언어모델과 Baseline Code를 제공하기 위해 만들어졌습니다.  
대회에 관한 자세한 설명은 [한국어 AI 언어모델 튜닝대회](https://challenge.enliple.com/)를 참고바랍니다.

Model의 Capacity가 커질수록 성능면에서는 이점이 있을 수 있으나 그만큼 고사양의 Machine이 필요합니다.  
그래서 Fine-tuning을 빠르게 수행할 수 있고 [Google Colab](https://colab.research.google.com/) 환경에서도 가능하도록 대회의 Pre-train Model은 Small Size로 진행하게 되었습니다.

Baseline Code는 해당 repository와 Google Colab 환경에서 제공하며 둘 중 편한것을 골라 사용하시면 됩니다.
* Colab Link:
  [Baseline Code in Colab](https://colab.research.google.com/drive/1n471tMpGoYlmoJpnSxTAZD_W2HKI1XnI)
  * 밑의 Model 및 Vocab을 다운받아 Google Drive에 업로드한뒤 Colab에서 Google Drive를 마운트하여 사용하시면 됩니다.

Model 및 Vocab은 다음 링크를 통해 다운로드 받을 수 있습니다.  
* [**Small Model  &  Vocab**](https://drive.google.com/open?id=13D9Fnnl0ra1qjPgtSWdp1-xIs6DfJ7Zg)


## Model Detail
해당 언어모델은 [Google BERT](https://github.com/google-research/bert) 및 [Huggingface Transformers](https://github.com/huggingface/transformers)를 참고하였으며 이를 Baseline으로 학습되었습니다.  
Baseline과 다른 점은 다음을 참고해주시면 감사하겠습니다.

|                 | Baseline                         | 대회제공모델                    |
|:----------------|:---------------------------------|:-------------------------------|
| MLM Strategy    | 15% random or whole word masking | n-gram masking                 |
| Additional Task | NSP(Next Sentence Prediction)    | SOP(Sentence Order Prediction) |
| Sub-word Level  | Space-level                      | Morpheme-Level                 |

* 또한 Google에서 공개한 BERT Small Size는 **Hidden:512, Layer:4, Attention-Head:8**로 세팅이 되어있지만 본 repository에서 제공하는 Small Size Model은 **Hidden:256, Layer:12, Attention-Head:8** 입니다.

## Train & Evaluation Scripts
* Train
```shell
python3 train.py \
  --output_dir=output \
  --checkpoint=pretrain_ckpt/bert_small_ckpt.bin \
  --model_config=data/bert_small.json \
  --train_file=data/KorQuAD_v1.0_train.json \
  --max_seq_length=512 \
  --max_query_length=96 \
  --max_answer_length=30 \
  --doc_stride=128 \
  --train_batch_size=16 \
  --learning_rate=5e-5 \
  --num_train_epochs=4.0 \
  --seed=42
```

* Evaluation
```shell
python3 eval.py \
  --checkpoint=output/korquad_3.bin \
  --output_dir=debug \
  --predict_file=data/KorQuAD_v1.0_dev.json \
  --max_seq_length=512 \
  --max_query_length=96 \
  --max_answer_length=30 \
  --doc_stride=128 \
  --train_batch_size=16 \
  --n_best_size=20 \
  --seed=42
```

* Result
```shell
{"exact_match": 78.6802909594735, "f1": 88.19766039994913}
```


## Reference
* [Google BERT](https://github.com/google-research/bert)
* [Huggingface Transformers](https://github.com/huggingface/transformers)
* [KorQuAD](https://korquad.github.io/)
---

* 추가적으로 궁금하신점은 해당 repo의 issue를 등록해주시거나 ekjeon@enliple.com으로 메일 주시면 답변 드리겠습니다.