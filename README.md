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
Since this is a relatively complex dataset, I wanted to analyze it to verify whether a traditional data analysis approach would be sufficiently good to predict and understand the patters of the data. Although there are other datasets related to the listing dataset, at this phase, I only focused on analyzing this dataset. Certainly, by directly including the other two datasets (calendar and reviews) we could gain more knowledge from the data. Finally, I used the CRISP-DM approach to investigate this dataset and answer business related questions.

## Summary of the results
At the beginning of the analysis, I provided four questions to drive data exploration and interpretation. The questions and a summary for each one can be found below.

1. What is the neighborhood with more listings?
The neighborhoods with more listings are: Jamaica Plain(343), South End (326), Back Bay (302), Fenway(290), Dorchester (269), and Allston (260).

2. Is there a relationship between the amount of listings and availability by neighborhood? 
<br>Those variables are negatively correlated (-0.35). However, the p-value was greater than 0.05 and the sample size for this analysis is quite small (n=25). Therefore, I would be skeptical to confirm that there is a significant correlation between the amount of listings and place availability.

<br>Still, those results are interesting. This trend shows that if a neighborhood has more listings, those places would have higher booking (less availability). This makes sense because, in theory, given the higher amount of listings, there is a higher demand of places in those neighborhoods. One exception is Dorchester that has many listings but fewer bookings than many other neighborhoods with less listings.

3. What is the vibe of each neighborhood based on the neighborhood overview?
<br>To answer this analysis, I performed a natural language processing analysis using the transformers module. The full analysis can be viewed in the notebook. In summary, the model did a good job in summarizing the owners overviews about the neighborhood, see this example: 
> Neighborhood: Roslindale - 42 reviews. Roslindale is a primarily residential neighborhood in the city of Boston. the neighborhood is well connected via public transportation to other neighborhoods. a farmer's market held in the neighborhood's center is held all summer on Saturday mornings. 

Some challenges about this analysis is further explored in the notebook.

4. Can we predict the price of Boston Airbnbs?
In order to answer this question, the data needed to be prepared as follows:

- The dollar sign was removed from all the currencies, making a column with floats instead of strings;
- Columns including some type of percentage were also changed to numeric floats (0-1);
- Columns including Booleans that were actually string (f or t) were changed to integers (0 or 1);
- Unnecessary qualitative columns were removed (id, host_id etc);
- Dummy variables were created for the relevant qualitative features, such as neighborhood and property type;
- The amenities column that each row was a list of strings were converted to the amount of amenities.

After this initial treatment, the NaNs were removed as follows:
- Every column including more than 50% of missing data were completely removed;
- In the Columns with very few NaNs (less than 0.5%), those rows with NaNs were dropped;

Subsequently, a hot deck imputation method was used to input data in the place of missing values. The KNNI module from sklearn.impute was used to do this task. The number of neighborhoods was set to 60, which is the square-root of the dataset length (rule of thumb).

After those procedures, the data were ready to be analyzed using machine learning techniques. 
Initially, using all the features and linear regression the model yielded a R^2 of 0.43. Next, the function selectKbest was used to only keep the features that increased the R^2. This analysis dropped 9 features but the model yielded a similar R^2. The model could predict 

In summary the model could predict 43% of the variance. I believe that by using more complex models and including the other datasets, this accuracy can be significantly improved. Still, given the limitations of the dataset, I believe that the model did a good job in general. 



## Contact

If you have questions or comments about the analysis, please contact me at: daniloarruda13@hotmail.com



[listing_dataset]: https://www.kaggle.com/datasets/airbnb/boston