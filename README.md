# DARE-ReChorus

## 项目说明

- 仅供中山大学机器学习大作业检查使用。

## 环境要求

- 没有严格的包版本要求，通常只要有对应的包就能运行。已经修改ReChorus框架中的过时Numpy语句（如`np.object`）。
- 原requirements.txt疑似存在问题，已将其改名为requirements_ReChorus.txt，不需要参考。
- 可先尝试根据**requirements.in**安装对应的包。若无法运行，可以使用命令`pip install -r requirements.txt`
  （该文件使用pip-compile工具生成）安装特定版本的包，本人测试时的Python版本为`3.12.3`。

## 运行命令

首先运行`cd src`，以下是训练时使用的命令：

### ML_1M CTR 数据集

1. DIN 模型
   `python main.py --model_name DIN --history_max 20 --lr 5e-4 --l2 1e-4 --dnn_layers "[512,64]" --att_layers "[64]" --dropout 0.5 --dataset ML_1MCTR --path ../data/ --num_neg 0 --batch_size 1024 --metric AUC,Log_loss --include_item_features 1 --include_situation_features 1 --model_mode CTR --loss_n BCE`
2. DIEN 模型
   `python main.py --model_name DIEN --lr 5e-4 --l2 1e-4 --history_max 20 --alpha_aux 0.5 --aux_hidden_layers "[64,64,64]" --fcn_hidden_layers "[256]" --evolving_gru_type AIGRU --dropout 0.2 --dataset ML_1MCTR --path ../data/ --num_neg 0 --batch_size 1024 --metric AUC,Log_loss --include_item_features 1 --include_situation_features 1 --model_mode CTR --loss_n BCE`
3. DARE 模型
   `python main.py --model_name DARE --history_max 20 --lr 5e-4 --l2 5e-5 --dnn_layers "[512,64]" --dropout 0.5 --dataset ML_1MCTR --path ../data/ --num_neg 0 --batch_size 1024 --metric AUC,Log_loss --include_item_features 1 --include_situation_features 1 --model_mode CTR --loss_n BCE`

### MIND_small CTR 数据集

1. DIN 模型
   `python main.py --model_name DIN --dropout 0.5 --lr 1e-4 --l2 5e-4 --history_max 20 --dataset MINDCTR --path ../data/ --num_neg 0 --batch_size 1024 --eval_batch_size 1024 --metric AUC,Log_loss --include_item_features 1 --include_situation_features 1 --model_mode CTR --att_layers "[32]" --dnn_layers "[128,32]" --loss_n BCE`
2. DIEN 模型
   `python main.py --model_name DIEN --lr 5e-4 --l2 1e-4 --history_max 20 --alpha_aux 0.1 --aux_hidden_layers "[64,64]" --fcn_hidden_layers "[256]" --evolving_gru_type AIGRU --dropout 0.2 --dataset MINDCTR --path ../data/ --num_neg 0 --batch_size 1024 --metric AUC,Log_loss --include_item_features 1 --include_situation_features 1 --model_mode CTR --loss_n BCE`
3. DARE 模型
   `python main.py --model_name DARE --history_max 20 --lr 2e-4 --l2 1e-4 --dnn_layers "[512,64]" --dropout 0.5 --dataset MINDCTR --path ../data/ --num_neg 0 --batch_size 1024 --metric AUC,Log_loss --include_item_features 1 --include_situation_features 1 --model_mode CTR --loss_n BCE`

## 其它说明

- 原ReChorus框架的代码基本无修改，仅修改了会导致运行报错的笔误（如变量名错误）。
- 由于MIND_large数据集过大，改为使用MIND_small数据集，对应的代码修改在`./data/MIND_small`目录中。
- 复现的DARE模型位于`./src/models/context_seq/DARE.py`。
- 其它细节详见报告。
