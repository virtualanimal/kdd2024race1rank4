# WhoIsWho-IND-KDD-2024 rank4

## Prerequisites

- Linux
- Python 3.10
- PyTorch 2.2.0+cu121

## Final Method

**Our final approach is to merge the results of the test set using the GCN model, the Xgboost machine learning model, and the llm model (ChatGLM) after fine tuning.**

| Method                                                       | AUC           |
| ------------------------------------------------------------ | ------------- |
| **GCN**                                                      | 0.7687(test)  |
| **Xgboost**（before using oagbert）                          | 0.7993(test)  |
| (**ChatGLM**(title5000_venue2500) + **ChatGLM**(title10000))/2 | 0.78790(test) |
| （GCN* 0.1+ Xgboost*  0.9）* 0.6 + (ChatGLM(title5000_venue2500)+ChatGLM(title10000))/2*0.4 | 0.8131(test)  |

## Getting Started

### Installation

Clone this repo.

```shell
git clone https://github.com/virtualanimal/kdd2024race1rank4.git
cd kdd2024race1rank4
```

For `Xgboost`,

```shell
pip install -r Xgboost/requirements.txt
```

For `GCN`,

```shell
pip install -r GCN/requirements.txt
```

For `ChatGLM`,

```shell
pip install -r ChatGLM/requirements.txt
```

## IND Dataset and DataProcess

The dataset can be downloaded from [BaiduPan](https://pan.baidu.com/s/1_CX50fRxou4riEHzn5UYKg?pwd=gvza) with password gvza, [Aliyun](https://open-data-set.oss-cn-beijing.aliyuncs.com/oag-benchmark/kddcup-2024/IND-WhoIsWho/IND-WhoIsWho.zip) or [DropBox](https://www.dropbox.com/scl/fi/o8du146aafl3vrb87tm45/IND-WhoIsWho.zip?rlkey=cg6tbubqo532hb1ljaz70tlxe&dl=1). Unzip the dataset and put files into `dataset/` directory.

+ DataProcess

  Before training, we should normalize paper info and extract feature from paper info(title, abstract, auther name, auther org, keywords, veneu and year),those feature can be used in GCN method (That means you should first put dataset in the right place and use Xgboost's pre three commands). you can check the detail in Xgboost's README file and  you can generate those feature (include scores and embeddings) by following Xgboost from step1 to step3。

  **This process may takes a lot of time, please be patinet. If you need,  we will upload our processed file later.**

+ Pretrained Model prepare

  We use three word embeddding tools(includeing [bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)、[sci-bert]([allenai/scibert_scivocab_uncased at main (huggingface.co)](https://huggingface.co/allenai/scibert_scivocab_uncased/tree/main)) and oag-bert, oag-bert can be donwload by tool cogdl(in Xgboost requirements.txt)), you should download those pretrained model and put them in `model/` directory(except oag-bert and remember change download model name to  <u>bge-small-en-v1.5</u> and <u>scibert</u> ). 

## Run Method for [KDD Cup 2024](https://www.biendata.xyz/competition/ind_kdd_2024/)

We provide three Method: [GCN](https://arxiv.org/abs/1609.02907), Xgboost, and [ChatGLM](https://arxiv.org/abs/2210.02414) [[Hugging Face\]](https://huggingface.co/THUDM/chatglm3-6b-32k). 

For `Xgboost`,

​	Do feature engineering and train xgboost model to predict results at 10 fold

```shell
cd Xgboost
# step1: Preprocessing Data
python norm_data.py
# step2: Embedding vector
python encode.py
# step3: Get features
python get_feature.py
# step4: predict
python predict.py

or bash run.sh
```

For `GCN`,

​	Build graph relational data, train and predict results using gcn model.

```shell
export CUDA_VISIBLE_DEVICES='?'  # specify which GPU(s) to be used
cd GCN
# as same as Xgboost pre three command 
#python norm_data.py
#python encode.py
#python get_feature.py
python build_graph.py 
bash train.sh #include train and predict
```

For `ChatGLM`,

Two fine-tuned ChatGLM checkpoint via Lora can be downloaded from  [ChatGLM-lora](https://drive.google.com/drive/folders/1YAbHMGZOq2PScc9c2NfEfTCmZXMRRNt6?usp=drive_link)

+ fineturn  with title(len=10000)

  ```shell
  export CUDA_VISIBLE_DEVICES='?'  # specify which GPU(s) to be used
  cd ChatGLM
  bash train.sh
  accelerate launch --num_processes 8 inference.py --lora_path your_lora_path --model_path your_model_path --pub_path  ../dataset/norm_pid_to_info_all.json --eval_path ../dataset/IND-test-public/ind_test_author_filter_public.json  # multi-GPU
  python inference.py --lora_path your_lora_checkpoint --model_path path_to_chatglm --pub_path ../dataset/norm_pid_to_info_all.json  --eval_path ../dataset/IND-test-public/ind_test_author_filter_public.json # single GPU
  ```

+ fineturn  with title(len=5000)+venue(len=2500)

  ```shell
  export CUDA_VISIBLE_DEVICES='?'  # specify which GPU(s) to be used
  cd ChatGLM
  bash train2.sh
  accelerate launch --num_processes 8 inference2.py --lora_path your_lora_path --model_path your_model_path --pub_path  ../dataset/norm_pid_to_info_all.json --eval_path ../dataset/IND-test-public/ind_test_author_filter_public.json  # multi-GPU
  python inference2.py --lora_path your_lora_checkpoint --model_path path_to_chatglm --pub_path ../dataset/norm_pid_to_info_all.json  --eval_path ../dataset/IND-test-public/ind_test_author_filter_public.json # single GPU
  ```

+ merge two method result

  ```shell
  cd ChatGLM
  python merge.py --first_json your_first_file_path --second_json your_second_file_path ----merge_llm_name your_merge_name
  ```
  

### Result

`All Model Merge`

```shell
python merge.py 
```

## File Struct

```shell
.
├── ChatGLM
│   ├── arguments.py
│   ├── configs
│   │   └── deepspeed.json
│   ├── finetune2.py
│   ├── finetune.py
│   ├── inference2.py
│   ├── inference.py
│   ├── merge.py
│   ├── metric.py
│   ├── output
│   ├── README.md
│   ├── requirements.txt
│   ├── train2.sh
│   ├── trainer.py
│   ├── train.sh
│   ├── utils2.py
│   └── utils.py
├── dataset
│   ├── embedding
│   ├── feature
│   ├── graph
│   └── result
├── GCN
│   ├── build_graph.py
│   ├── encode.py
│   ├── get_feature.py
│   ├── models.py
│   ├── norm_data.py
│   ├── README.md
│   ├── requirements.txt
│   ├── train.py
│   └── train.sh
├── merge.py
├── model
├── README.md
└── Xgboost
    ├── encode.py
    ├── get_feature.py
    ├── norm_data.py
    ├── predict.py
    ├── README.md
    ├── requirements.txt
    └── run.sh
```

and in `dataset/`

<img src="img/image-20240613143223232.png" alt="image-20240613143223232" style="zoom: 67%;" />
