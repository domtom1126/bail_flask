import pandas as pd
import joblib
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('all_orc.csv', usecols=['charge_orc', 'prior_cases', 'bond_amt', 'judge', 'race'])
df.dropna(inplace=True)

df['plea_orcs'] = df['charge_orc'].str.split('(').str[0]

plea_list = list(df['plea_orcs'].unique())
# plea_list.sort(key=float)

race_list = list(df['race'].unique())

le = LabelEncoder()
le2 = LabelEncoder()
le3 = LabelEncoder()

df.charge_orc = le.fit_transform(df.charge_orc.astype(str))
df.judge = le2.fit_transform(df.judge)
df.race = le3.fit_transform(df.race)

X = df[['plea_orcs', 'race', 'prior_cases']]
y = df[['bond_amt']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

reg = DecisionTreeRegressor()
reg.fit(X_train, y_train)

# joblib.dump(reg, 'bail_model')
