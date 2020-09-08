

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

# BERT를 활용한 영어 다중 감성 분류기

현재까지의 한국어 자연어의 감성 분류는 대부분이 긍정-부정의 이진Binary 분류 (=극성Polarity 분류)에 한정되어 있습니다.
하지만 실제 사람의 감정은 단순히 긍정-부정보다 다양한 분류를 가집니다.

이번 시도에서는 영어 자연어 처리를 위한 [BERT](https://github.com/google-research/bert) 모델을 활용했습니다.  
한국어에 비해 영어 자연어 처리는 더 많은 자료와 더 높은 성능을 나타내기에 한국어->영어 번역 과정을 중간에 넣어 BERT를 사용하고자 하였습니다.

참고 자료:
Google-research [BERT FineTuning with Cloud TPUs](https://colab.research.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb)
Renu Khandelwal [Multi-class Sentiment Analysis using BERT](https://towardsdatascience.com/multi-class-sentiment-analysis-using-bert-86657a2af156?gi=c537e046d9bc)

### Dataset
[EmoInt](https://saifmohammad.com/WebPages/EmotionIntensity-SharedTask.html)의 영어 감정 레이블링 데이터를 사용하였습니다.


### 결과

문장 동일성 평가 문제를 위해 finetuning된 모델을 다중 감성 분류 모델로 바꾸는 데에 실패하였습니다.

### 후기

data_loader부터 classifier 부분까지, 세세한 부분들에서 수정해야할 부분들이 계속해서 발견됨.  
모델을 직접 만들지 않고 다른 사람이 짜놓은 모델을 가져다가 내 문제를 해결하기 위해 고치는 것도 상당한 품이 든다는 것을 깨달았다.


