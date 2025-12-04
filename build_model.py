import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import GradientBoostingRegressor
import sklearn

df = pd.read_csv("honda_toyota_ca.csv")

cols = df.columns.tolist()[1:]

X = df[cols]
y = df["price"]

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


t_f = [i for i in X.dtypes == "object"]
dict_cols = dict(zip([c for c in X], t_f))
text_cols = [key for key, value in dict_cols.items() if value == True]

ore_transformer = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

preprocessor = ColumnTransformer(
    transformers=[("s_cols", ore_transformer, text_cols)], remainder="passthrough"
)

pipe = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("gbr", GradientBoostingRegressor(random_state=42)),
    ]
)

pipe.fit(x_train, y_train)

dump(pipe, "trained_model.joblib")
print(f"✅ Model trained with Scikit-learn version: {sklearn.__version__}")
