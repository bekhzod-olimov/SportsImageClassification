# Sports Image Classification

This repository contains a deep learning (DL)-based artificial intelligence (AI) image classification model training to classify different sports types. The AI model used for the classification task is RexNet ([paper](https://arxiv.org/pdf/2007.00992.pdf) and [code](https://github.com/clovaai/rexnet)) and the dataset for training is [Fruits 100 Dataset](https://www.kaggle.com/datasets/sidharkal/sports-image-classification). The project in [Kaggle](https://www.kaggle.com/) can be found [here](https://www.kaggle.com/code/killa92/sports-classification-pytorch-98-accuracy). The models can be trained using two different frameworks ([PyTorch](https://pytorch.org/) and [PyTorch Lightning](https://lightning.ai/)).

# Manual on how to use the repo:

1. Clone the repo to your local machine using terminal via the following script:

```python
git clone https://github.com/bekhzod-olimov/SportsImageClassification.git
```

2. Create conda environment from yml file using the following script:

a) Create a virtual environment using txt file:

- Create a virtual environment:

```python
conda create -n speed python=3.9
```

- Activate the environment using the following command:

```python
conda activate speed
```

- Install libraries from the text file:

```python
pip install -r requirements.txt
```

b) Create a virtual environment using yml file:

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

a) PyTorch training:

```python
python main.py --root PATH_TO_THE_DATA --batch_size = 64 device = "cuda:0" --train_framework "py"
```
The training parameters can be changed using the following information:

![image](https://github.com/bekhzod-olimov/SportsImageClassification/assets/50166164/d6ef5b40-b792-4654-ae23-f1259a01c7f7)

The training process progress:

![image](https://github.com/bekhzod-olimov/SportsImageClassification/assets/50166164/7a142c81-9f78-4348-a1ff-8efc5d01e7b9)

b) PyTorch Lightning training:

```python
python main.py --root PATH_TO_THE_DATA --batch_size = 64 device = "cuda:0" --train_framework "pl"
```

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

![image](https://github.com/bekhzod-olimov/SportsImageClassification/assets/50166164/4885b0a0-91f6-452f-b9d2-39e336750d20)

7. Inference Results (Predictions):

![sports_preds](https://github.com/bekhzod-olimov/SportsImageClassification/assets/50166164/317e28a8-1e85-447d-8ed4-f3b41c880ca7)

8. Inference Results (GradCAM):
   
![sports_gradcam](https://github.com/bekhzod-olimov/SportsImageClassification/assets/50166164/b3007af6-4880-48b4-b490-6462a545402f)

9. Run Demo

```python
streamlit run demo.py
```

10. [Demo](http://218.38.14.21:8501/): 
![image](https://github.com/bekhzod-olimov/SportsImageClassification/assets/50166164/5c04c5ef-5f2f-4c42-8aa8-b5ebe295eeed)
