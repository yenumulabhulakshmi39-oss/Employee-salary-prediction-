#  Import libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

#  Sample dataset without nulls or blanks
data = {
    'Experience': [1, 3, 5, 7, 9, 2, 4, 6, 8, 10],
    'Education': ['Bachelor', 'Bachelor', 'Master', 'Master', 'PhD',
                  'Bachelor', 'Master', 'Master', 'PhD', 'PhD'],
    'JobTitle': ['Developer', 'Tester', 'Analyst', 'Manager', 'Director',
                 'Developer', 'Analyst', 'Manager', 'Director', 'Director'],
    'Salary': [300000, 400000, 600000, 900000, 1500000,
               320000, 610000, 950000, 1450000, 1550000]
}

df = pd.DataFrame(data)
# check for null or blanks
assert not df.isnull().values.any(), "Dataset contains null values!"
assert not (df.astype(str).apply(lambda x: x.str.strip()) == '').any().any(), "Dataset contains blank strings!"

# Encode categorical features
le_edu = LabelEncoder()
le_job = LabelEncoder()
df['Education'] = le_edu.fit_transform(df['Education'])     # Bachelor=0, Master=1, PhD=2
df['JobTitle'] = le_job.fit_transform(df['JobTitle'])       # Developer=0, etc.

#  Features and Target
X = df[['Experience', 'Education', 'JobTitle']]
y = df['Salary']

#  Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train KNN model
model = KNeighborsRegressor(n_neighbors=3)
model.fit(X_train, y_train)

#  Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
