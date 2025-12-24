import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from train_utils import DATA_FILE_PATH, MODEL_PATH, MODEL_DIR

print("libraries imported succesfully.")

df = (
    pd.read_csv(DATA_FILE_PATH)
    .drop_duplicates()
    .drop(columns=["name", "model", "edition"])
)
print("Data read succesfully")

X = df.drop(columns="selling_price")
y = df.selling_price.copy()

print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



num_cols = X_train.select_dtypes(include="number").columns.tolist()
cat_cols = [col for col in X_train.columns if col not in num_cols]

num_pipe = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

cat_pipe = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)]
)


param_grids = [
    {
        "reg": [RandomForestRegressor(random_state=42)],
        "reg__n_estimators": [10, 50, 100],
        "reg__max_depth": [5, 10, None],
    },
    {
        "reg": [GradientBoostingRegressor(random_state=42)],
        "reg__n_estimators": [10, 50, 100],
        "reg__learning_rate": [0.01, 0.1, 0.3],
    },
    {"reg": [Ridge()], "reg__alpha": [0.1, 1.0, 10.0]},
]

# Create base pipeline
base_pipeline = Pipeline(
    steps=[
        ("pre", preprocessor),
        ("reg", RandomForestRegressor()),  # Dummy, will be replaced
    ]
)

print("Starting model training.")

# Grid search over all parameter grids
grid_search = GridSearchCV(
    base_pipeline, param_grids, cv=5, scoring="r2", refit=True, verbose=1, n_jobs=-1
)

model = grid_search.fit(X_train, y_train)

print(f"Best model: {grid_search.best_estimator_}")
print(f"Best score: {grid_search.best_score_:.4f}")

os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(model, MODEL_PATH)
print(f"Model saved to: {MODEL_PATH}")
