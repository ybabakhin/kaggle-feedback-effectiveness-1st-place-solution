## Feedback Prize - Predicting Effective Arguments: 1st place solution code

It's 1st place solution to Kaggle competition: Feedback Prize - Predicting Effective Arguments: https://www.kaggle.com/competitions/feedback-prize-effectiveness/

This repo contains the code for training the models, while the solution writeup is available here: https://www.kaggle.com/competitions/feedback-prize-effectiveness/discussion/347536

### Environment

To setup the environment:
* Install python3.8
* Install `requirements.txt` in the fresh python environment

## Main LB solution

### Training

* Download additional training data from https://www.kaggle.com/datasets/ybabakhin/feedback-competition-team-hydrogen-data and extract it to `./data` directory
* This data can also be generated by the notebooks in the `./notebooks` directory
* To train the first level models, run: `./train.sh`
* To train the second level models, run the notebooks in the `./notebooks/second_stage/` directory

### Inference

Final inference kernel is available here: https://www.kaggle.com/code/ybabakhin/team-hydrogen-1st-place

## Efficiency LB solution

The efficiency solution is described here: https://www.kaggle.com/competitions/feedback-prize-effectiveness/discussion/347537

### Training

To train an efficiency model run: `python train.py -C yaml/efficiency_model.yaml`

### Inference

Link to the inference kernel: https://www.kaggle.com/code/philippsinger/team-hydrogen-efficiency-prize-1st-place
