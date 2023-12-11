# Sports Image Classification

This repository contains a deep learning (DL)-based artificial intelligence (AI) image classification model training to classify different sports types. The AI model used for the classification task is RexNet ([paper](https://arxiv.org/pdf/2007.00992.pdf) and [code](https://github.com/clovaai/rexnet)) and the dataset for training is [Sports Image Classification Dataset](https://www.kaggle.com/datasets/sidharkal/sports-image-classification). The project in [Kaggle](https://www.kaggle.com/) can be found [here](https://www.kaggle.com/code/killa92/sports-classification-pytorch-98-accuracy).

# Manual on how to use the repo:

1. Clone the repo to your local machine using terminal via the following script:

```python
git clone https://github.com/bekhzod-olimov/SportsImageClassification.git
```

2. Create conda environment from yml file using the following script:
```python
conda env create -f environment.yml
```
Then activate the environment using the following command:
```python
conda activate speed
```

3. Data Visualization

![image](https://github.com/bekhzod-olimov/SportsImageClassification/assets/50166164/6035063c-e7dd-4fd5-a388-8a9251d2de05)

4. Train the AI model using the following script:
```python
python main.py --root PATH_TO_THE_DATA --batch_size = 64 device = "cuda:0"
```
The training parameters can be changed using the following information:

![image](https://github.com/bekhzod-olimov/SportsImageClassification/assets/50166164/807af4f1-a7f3-410a-8bff-89c2984c9b27)

The training process progress:

![image](https://github.com/bekhzod-olimov/JellyfishClassifier/assets/50166164/ca68a5af-bef1-4b41-9412-648c2f4942c3)

5. Learning curves:
   
Use [DrawLearningCurves](https://github.com/bekhzod-olimov/SportsImageClassification/blob/266aa4f15aef5ea5887e228c1b85ab7c4627047f/main.py#L56) class to plot and save learning curves.

* Train and validation loss curves:
  
![loss_learning_curves](https://github.com/bekhzod-olimov/SportsImageClassification/assets/50166164/f1d20ff2-62bf-4514-af00-e5b5a225f57a)

* Train and validation accuracy curves:
  
![acc_learning_curves](https://github.com/bekhzod-olimov/SportsImageClassification/assets/50166164/ca0b9c2f-94cf-4d1e-bb53-c39d9aeb335b)

6. Conduct inference using the trained model:
```python
python inference.py --root PATH_TO_THE_DATA --batch_size = 64 device = "cuda:0"
```

The inference progress:

![image](https://github.com/bekhzod-olimov/JellyfishClassifier/assets/50166164/8fee0d75-c43c-4b85-9fcd-9a285a4cdf4a)

7. Inference Results (Predictions):

![brain_preds](https://github.com/bekhzod-olimov/SportsBallClassification/assets/50166164/7a8ffd57-4f66-4383-a5f5-06d032dbc27f)
