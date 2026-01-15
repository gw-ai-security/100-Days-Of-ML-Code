# Data PreProcessing
<p align="center">
  <img src="https://github.com/Avik-Jain/100-Days-Of-ML-Code/blob/master/Info-graphs/Day%201.jpg">
</p>

As shown in the infograph we will break down data preprocessing in 6 essential steps.
Get the dataset from [here](https://github.com/Avik-Jain/100-Days-Of-ML-Code/tree/master/datasets) that is used in this example

## Setup in VS Code
1) Install Python 3.x and ensure it is on PATH.
2) Create and activate a venv in the repo root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3) Install dependencies in the same environment:

```powershell
python -m pip install numpy pandas scikit-learn jupyter ipykernel
```

Then select the `.venv` kernel in VS Code and run the cells.

## Step 1: Importing the libraries
```Python
import numpy as np
import pandas as pd
```
## Step 2: Importing dataset
```python
from pathlib import Path
dataset = pd.read_csv(Path("datasets") / "Data.csv")
X = dataset.iloc[ : , :-1].values
Y = dataset.iloc[ : , 3].values
```
## Step 3: Handling the missing data
```python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = "mean")
X[ : , 1:3] = imputer.fit_transform(X[ : , 1:3])
```
## Step 4: Encoding categorical data
```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(transformers = [('country', OneHotEncoder(), [0])], remainder = 'passthrough')
X = ct.fit_transform(X)
```
### Creating a dummy variable
```python
labelencoder_Y = LabelEncoder()
Y =  labelencoder_Y.fit_transform(Y)
```
## Step 5: Splitting the datasets into training sets and Test sets 
```python
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X , Y , test_size = 0.2, random_state = 0)
```

## Step 6: Feature Scaling
```python
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
```
### Done :smile:
