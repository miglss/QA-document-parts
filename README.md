# Question answering in legal contracts (RU) 💬
Assignment completed as part of the selection process for the DS internship at Kontur
## Stack of technologies 🏗
- Python 🐍
- Transformers 🤗
- Wandb 📊
## Task description 📋
The task is to find a piece of text in the contract that corresponds to one of the queries: "обеспечение гарантийных обязательств" or "обеспечение исполнения контракта". 

A detailed description of the task can be found in [task_description.md](https://github.com/miglss/QA-document-parts/blob/main/task_description.md)
## Proposed solution 💡
For this task mdeberta-v3-base-squad2 model from 🤗 was fine-tuned. Custom tokenizer was also trained on a new corpus.

Hyperparameters optimization was performed using Wandb sweep's: 

<img src="images/hyperparameter_optim.png" alt="map" width="800"/>

After that, model was trained for 30 epochs using best set of hyperparameters

Although a smaller number of epochs could have been avoided, as can be seen from the losses graph:

<img src="images/train_val_loss.png" alt="map" width="800"/>

Quality on validation set during training:

<img src="images/metrics.png" alt="map" width="800"/>

Final quality on test:
- exact score: 84.44444444444444
- f1-score: 97.47689267517949

## How to improve 🔨
1. Longer training as well as hyperparameters optimization can significantly improve the quality of the final model
2. The U-net architecture is rather obsolete and rarely used in modern solutions. The use of newer architectures will give much more sustainable results
