# Exploring Boston's Airbnb Trends: Insights from data
This is a python project, in which I analyze the listing dataset [here][listing_dataset] of Boston Airbnbs. The analysis includes description of the listings, Natural Language Processing to summarize the neighborhood overviews, and machine learning to predict prices.

## Table of Contents

- [Installation](#installation)
- [Repository Files](#repository-files)
- [Motivation](#motivation)
- [Summary of the Results](#summary-of-the-results)
- [Contact](#contact)


## Installation
All the needed libraries can be directly imported using the "requirements.txt" file, using the following code in the terminal:
```python
pip install -r requirements.txt
```
## Repository Files
1. Bonston_AIRbnb.ipynb: This is a Jupyter Notebook with all the analysis. In order to run the notebook, the directory of the dataset should be appropriately changed. 
2. listing.csv: This is the dataset used for the analysis. It includes variables related the Boston Airbnb listings, such as price, type (house, apt), review scores, among others.
3. requirements.txt: This is the txt file with all the required libraries to run the Jupyter Notebook. The imported modules are:

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import r2_score
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import KNNImputer
import textwrap
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
```
4. README.md: This is the markdown readme file containing all the relevant information about the files and data analysis.

## Motivation
Since this is a relatively complex dataset, I wanted to analyze it to verify whether a traditional data analysis approach would be sufficiently good to predict and understand the patters of the data. Although there are other datasets related to the listing dataset, at this phase, I only focused on analyzing this dataset. Certainly, by directly including the other two datasets (calendar and reviews) we could gain more knowledge from the data. 

## Summary of the results
At the beginning of the analysis, I provided four questions to drive data exploration and interpretation. The questions and a summary for each one can be found below.

1. What is the neighborhood with more listings?

2. Is there a relationship between the amount of listings and availability by neighborhood? 
3. What is the vibe of each neighborhood based on the neighborhood overview?
4. Can we predict the price of Boston Airbnbs?

## Contact





[listing_dataset]: https://www.kaggle.com/datasets/airbnb/boston