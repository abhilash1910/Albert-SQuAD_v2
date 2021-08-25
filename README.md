# Albert-SQuAD_v2




<img src="https://huggingface.co/front/assets/huggingface_logo.svg">



This Notebook contains set of instructions how to train [Albert Large](https://huggingface.co/transformers/model_doc/albert.html) from Huggingface in Google Colab.
Training is done on the [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) dataset. The model can be accessed via [HuggingFace](https://huggingface.co/abhilash1910/albert-squad-v2):


```python

from transformers import AutoTokenizer,AutoModelForQuestionAnswering
from transformers import pipeline
model=AutoModelForQuestionAnswering.from_pretrained('./results/checkpoint-19500/')
tokenizer=AutoTokenizer.from_pretrained('./results/checkpoint-19500/')
nlp_QA=pipeline('question-answering',model=model,tokenizer=tokenizer)
QA_inp={
    'question': 'How many parameters does Bert large have?',
    'context': 'Bert large is really big... it has 24 layers, for a total of 340M parameters.Altogether it is 1.34 GB so expect it to take a couple minutes to download to your Colab instance.'
}
result=nlp_QA(QA_inp)
result
```

The result is:

```bash

{'answer': '340M', 'end': 65, 'score': 0.14847151935100555, 'start': 61}
 ```

Model Config:

```bash

Model config AlbertConfig {
  "_name_or_path": "albert-large-v1",
  "architectures": [
    "AlbertForQuestionAnswering"
  ],
  "attention_probs_dropout_prob": 0.1,
  "bos_token_id": 2,
  "classifier_dropout_prob": 0.1,
  "down_scale_factor": 1,
  "embedding_size": 128,
  "eos_token_id": 3,
  "gap_size": 0,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 1024,
  "initializer_range": 0.02,
  "inner_group_num": 1,
  "intermediate_size": 4096,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "albert",
  "net_structure_type": 0,
  "num_attention_heads": 16,
  "num_hidden_groups": 1,
  "num_hidden_layers": 24,
  "num_memory_blocks": 0,
  "pad_token_id": 0,
  "position_embedding_type": "absolute",
  "torch_dtype": "float32",
  "transformers_version": "4.9.2",
  "type_vocab_size": 2,
  "vocab_size": 30000
}
```

## Demo

[Kaggle](https://www.kaggle.com/abhilash1910/training-albertqa)

