

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

# KoBERT를 활용한 한국어 다중 감성 분류기

BERT 모델을 사용한 한국어 자연어의 감성 분류는 대부분이 긍정-부정의 이진Binary 분류 (=극성Polarity 분류)에 한정되어 있습니다.
하지만 실제 사람의 감정은 단순히 긍정-부정보다 다양한 분류를 가집니다.

이를 위해 SKTBrain에서 개발한 pretrained [KoBERT](https://github.com/SKTBrain/KoBERT)를 다중 감성 분류에 적용해 보았습니다.

### Requirements

* Python >= 3.6
* konlpy >= 0.5.2
* PyTorch >= 1.1.0
* MXNet >= 1.4.0
* gluonnlp >= 0.6.0
* sentencepiece >= 0.1.6
* onnxruntime >= 0.3.0
* transformers >= 2.1.1

### Dataset
한국어 감정 데이터는 [AI 허브](http://www.aihub.or.kr/)의 오픈 데이터 "한국어_단발성_대화_데이터"를 사용하였습니다.

데이터 형태 
|Sentence|Emotion|
|------|:---:|
|아~ 정말 어떻게 해야되죠..1|공포|
| 혼자 웃다가 울다가 조울증인가싶기두해여..|공포|
| 월욜날연락주신댔는데지금불안해서미치겠어요ㅜㅜ|공포|

총 38954 개의 문장과 문장 각각에 해당하는 감정 7가지 {공포, 놀람, 분노, 슬픔, 중립, 행복, 혐오} 중 하나가 레이블 되어있습니다.

이 데이터를 전처리한 뒤 모델에 투입하였습니다.


### Architecture

SKTBrain의 [koBERT](https://github.com/SKTBrain/KoBERT)의 Tokenizer와 pretrained BERT 모델을 그대로 사용했습니다

```python
predefined_args = {
        'attention_cell': 'multi_head',
        'num_layers': 12,
        'units': 768,
        'hidden_size': 3072,
        'max_length': 512,
        'num_heads': 12,
        'scaled': True,
        'dropout': 0.1,
        'use_residual': True,
        'embed_size': 768,
        'embed_dropout': 0.1,
        'token_type_vocab_size': 2,
        'word_embed': None,
    }
```

### 사용 환경
구글 Colab의 GPU 런타임 유형을 사용하였습니다.  


### 결과
평가 Metric은 단순 Accuracy를 사용하였습니다.

학습 및 평가 결과 
|Model|학습 정확도|평가 정확도|
|------|:---:|:---:|
|KoBERT|0.598|0.569|

### 후기
- 학습 시 Epoch 3회만에 과적합이 일어났습니다. 데이터가 더 필요합니다.
- 2020년 7월 기준 다중 감성 분류 SOTA가 0.6 즈음인 것을 감안할 때, 생각보다 좋은 분류 성능을 보였습니다.
- 하지만 모든 domain에 대해 적용할만큼 일반화되지 않았다는 한계가 있습니다
- 한국어 감정이 레이블된 데이터가 더 많이 필요합니다.


