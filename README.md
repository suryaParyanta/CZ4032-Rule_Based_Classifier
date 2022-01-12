# CZ4032-Rule_Based_Classifier

This project is to implement model called Classification Based on Associations (CBA).
This model consists of two parts, *rule generator* (called CBA-RG) which uses Apriori algorithm for mining class association rules, and 
*classifier builder* (called CBA-CB) that builds classifier based on class association rules generated on previous part.

In addition, the improvement of CBA model, called Classification Based on Multiple Association Rules (CMAR) is implemented.
Similar to CBA, it is also consists of two parts, *rule generator* which is implemented using a more efficient algorithm FP-Growth, and
*classifier builder* that makes decisions through multiple groups of class association rules.

The paper can be found here:
1. CBA: [Integrating Classification and Association Rule Mining](https://www.aaai.org/Papers/KDD/1998/KDD98-012.pdf)
2. CMAR: [Accurate and Efficient Classification Based on Multiple Class-Association Rules](http://hanj.cs.illinois.edu/pdf/cmar01.pdf)

## Environment Setup

Download or clone this repository.
```bash
git clone https://github.com/suryaParyanta/CZ4032-Rule_Based_Classifier.git
```

Install the packages by running the following command:
```bash
pip install -r requirements.txt
```

If you do not have Jupyter Notebook on your PC, you can install it using pip:
```bash
pip install jupyter
```

## Getting Started

To evaluate the model performance, you can run the following command:
```bash
# MODEL_NAME choices: [CBA, CMAR]
# DATASET_NAME choices: [iris, wine, titanic, breast_cancer]
# For more information, you can run: python evaluate_model.py --help
python evaluate_model.py --model MODEL_NAME --dataset DATASET_NAME
```

## Training using Custom Dataset

To train with custom dataset, you need to:
1. Put your custom dataset on `datasets/raw` folder.
2. Rename the target class column with `class`.
3. Pre-processed the custom dataset so that it only contains categorical values. Equal-width and equal-frequency binning method is provided in the notebook.
4. Save the processed dataset into `datasets/processed` folder. The dataset filename should be in form of: `DATASET_NAME_processed.csv`.
5. Add the `DATASET_NAME` into list of choices found in `evaluate_model.py`

## Results

Classification accuracy:

Dataset Name  | CBA        | CMAR
:-----------: | :--------: | :--: 
Iris          | 96%        | 96% 
Wine          | 97.22%     | 97.75%
Breast Cancer | 93.5%      | 93.5%
Titanic       | 80.7%      | 81.04%
**Average**   | **91.86%** | **92.07%**


Average runtime speed per 10-fold cross validation (in seconds):

Dataset Name  | CBA        | CMAR
:-----------: | :--------: | :--: 
Iris          | 0.01 s     | 0.02 s 
Wine          | 3.93 s     | 0.21 s
Breast Cancer | 15.34 s    | 0.96 s
Titanic       | 7.42 s     | 0.56 s
**Average**   | **6.68 s** | **0.44 s**
