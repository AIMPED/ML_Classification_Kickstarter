import numpy as np
import pandas as pd
import pickle

# Scikit Learn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from datetime import datetime

RSEED = 42

df = pd.read_csv('data/Kickstarter_preprocessed.csv')

print('Data is read in')

# Deleting the first column (bears no information)
df.drop(['Unnamed: 0'],axis=1,inplace=True);

# Rename some columns to have more meaningful names
df.rename(columns={'name_category':'category_sub',
    'slug_category':'category','blurb':'description'},inplace=True)


# Time conversion
time_cols = ['created_at','deadline','state_changed_at','launched_at']
df[time_cols] = df[time_cols].applymap(lambda x: datetime.utcfromtimestamp(x))

df['description_length'] = df['description'].apply(lambda x: len(str(x).split()))

df = df.eval('usd_goal = static_usd_rate * goal')

# Calculating the duration of project
df['duration'] = df['deadline'] - df['launched_at']
df['duration_days']=df['duration'].dt.days

# Start year and month of the projects
df['start_month']= df['launched_at'].dt.month
df['start_year']= df['launched_at'].dt.year

# Splitting the text in column category, keeping only the left part of the string --> main category
df.category = df.category.apply(lambda x: x.split('/')[0])

# change to lower case string
df.category_sub = df.category_sub.str.lower()

categorical_features = [
    'currency', 
    'country', 
    'staff_pick', 
    'category', 
    'category_sub',
    'start_month'
]

# Convert strings and numbers to categories
df[categorical_features] = df[categorical_features].apply(lambda x: x.astype('category'))
# Convert strings to numbers
df[categorical_features] = df[categorical_features].apply(lambda x: x.cat.codes)
# Convert numbers to categories
df[categorical_features] = df[categorical_features].apply(lambda x: x.astype('category'))

features = ['description_length','duration_days','usd_goal','country','staff_pick','category','category_sub','start_month']
target = ['state']

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

print('Data cleaning and train-test split is done')

print ('Train set shape:', X_train.shape, y_train.shape)
print ('Test set shape: ', X_test.shape , y_test.shape)

# These are the parameters found via the grid search
rfc = RandomForestClassifier(max_depth=20,
                             max_features=None, 
                             min_samples_split=5, 
                             n_jobs=-1)
rfc.fit(X_train,y_train)

print('Model is fitted, saving model now')

filename = 'optimized_random_forest_model.pickle'
with open(filename, 'wb') as file:
    pickle.dump(rfc, file)

