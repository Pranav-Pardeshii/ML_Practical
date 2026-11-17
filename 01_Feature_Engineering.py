# Import required libraries
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load data
df = pd.read_csv('titanic.csv')
print("Missing Before:", df.isnull().sum()[df.isnull().sum()>0])

# Keep only useful cols
cols_to_drop = ['survived','class','who','adult_male','deck','embark_town','alive','alone']
X = df.drop(columns=cols_to_drop)

# Seperate numerical and categorical cols
num_cols = ['age','fare','sibsp','parch']
cat_cols = ['pclass','sex','embarked']

# Fix num_cols
imputer = SimpleImputer(strategy='median')
num_data = imputer.fit_transform(X[num_cols])

# Scale data
scaler = StandardScaler()
num_data = scaler.fit_transform(num_data)

# Fix cat_cols 
imputer = SimpleImputer(strategy='most_frequent')
cat_data = imputer.fit_transform(X[cat_cols])

# Encode cat_data
encoder = OneHotEncoder(sparse_output=False)
cat_data = encoder.fit_transform(cat_data)

# Concatenate both numerical and categorical datasets
X_final = pd.concat([pd.DataFrame(num_data), pd.DataFrame(cat_data)], axis=1)

# Print final shape and missing values
print("FInal shape", X_final.shape)
print("Missing values after", X_final.isnull().sum().sum())