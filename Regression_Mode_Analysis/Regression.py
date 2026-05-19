# 1. Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

# 2. Load Training Data
train_path = "/Users/tinal/Documents/6409/A1_competition_train.csv"
test_path  = "/Users/tinal/Documents/6409/A1_competition_test.csv"

df = pd.read_csv(train_path)

print(df.info())
print(df.isna().sum())

'''
None
row_id         0
mode           0
x              0
category       0
sensor_id      0
timestamp      0
note         130
y              9
'''

# 3. Data Cleaning
df = df.drop(columns=["note", "row_id"])
df = df.dropna(subset=["y"])

# 4. EDA
# Distribution of y
plt.figure()
plt.hist(df["y"].dropna(), bins=30)
plt.xlabel("y")
plt.ylabel("Frequency")
plt.title("Distribution of target variable y")
plt.show()

# Distribution of x
plt.figure()
plt.hist(df["x"], bins=30)
plt.xlabel("x")
plt.ylabel("Frequency")
plt.title("Distribution of feature x")
plt.show()

# x vs y by mode
plt.figure(figsize=(8,6))

for m in sorted(df["mode"].dropna().unique()):
    subset = df[df["mode"] == m]
    plt.scatter(subset["x"], subset["y"], s=20, label=f"mode {int(m)}")

plt.xlabel("x")
plt.ylabel("y")
plt.title("x vs y by mode")
plt.legend()
plt.show()

# 5. Feature Engineering
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.dropna(subset=["timestamp"])

df["hour"] = df["timestamp"].dt.hour
df["day"] = df["timestamp"].dt.day
df["weekday"] = df["timestamp"].dt.weekday
df["month"] = df["timestamp"].dt.month
df = df.drop(columns=["timestamp"])

df["sensor_id"] = df["sensor_id"].astype(str)
df["mode"] = df["mode"].astype(int).astype(str)
df["category"] = df["category"].astype(str)

features = ["x", "hour", "day", "weekday", "month"]

# 6. Train / Validation Split
X = df[features + ["mode", "category"]]
y = df["y"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

train_df = X_train.copy()
train_df["y"] = y_train

val_df = X_val.copy()
val_df["y"] = y_val

#Training Set: x vs y Plot by Mode
modes = sorted(train_df["mode"].unique())
n = len(modes)

cols = 3
rows = math.ceil(n / cols)

fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows), sharex=True, sharey=True)
axes = axes.flatten()

for i, m in enumerate(modes):
    sub = train_df[train_df["mode"] == m]

    axes[i].scatter(sub["x"], sub["y"], alpha=0.6)
    axes[i].set_title(f"Mode {m}: y vs x")
    axes[i].set_xlabel("x")
    axes[i].set_ylabel("y")

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.suptitle("Training Set: x vs y by Mode", fontsize=14)
plt.tight_layout()
plt.show()

# 5. Train models by mode
models = {}
cluster_models = {}
mode6_models = {}

for m in train_df["mode"].unique():
    df_m = train_df[train_df["mode"] == m]
    Xm = df_m[features]
    ym = df_m["y"]

    # mode 1,4 -> Linear Regression
    if m in ["1", "4"]:
        lr = LinearRegression()
        lr.fit(Xm, ym)
        models[m] = lr

    # mode 2 -> Polynomial Regression
    elif m == "2":
        poly_model = Pipeline([
            ("poly", PolynomialFeatures(degree=5, include_bias=False)),
            ("lr", LinearRegression())
        ])
        poly_model.fit(df_m[["x"]], ym)
        models[m] = poly_model

    # mode 3 -> KMeans
    elif m == "3":
        kmeans = KMeans(n_clusters=2, random_state=42)
        labels = kmeans.fit_predict(df_m[["x"]]) 

        cluster_means = {}
        for c in [0, 1]:
            cluster_means[c] = ym[labels == c].mean()

        cluster_models[m] = {
            "kmeans": kmeans,
            "means": cluster_means
        }

    # mode 5 -> Polynomial Regression
    elif m == "5":
        poly_model = Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("lr", LinearRegression())
        ])
        poly_model.fit(Xm, ym)
        models[m] = poly_model

    # mode 6 -> split by category
    elif m == "6":
        df_c0 = df_m[df_m["category"] == "0"]
        df_c1 = df_m[df_m["category"] == "1"]

        lr_c0 = LinearRegression()
        lr_c1 = LinearRegression()

        lr_c0.fit(df_c0[features], df_c0["y"])
        lr_c1.fit(df_c1[features], df_c1["y"])

        mode6_models[m] = {
            "c0": lr_c0,
            "c1": lr_c1
        }

# 6. Prediction function
def predict_row(row):
    m = row["mode"]

    if m == "2":
        X_row = pd.DataFrame([row[["x"]]])
        return models[m].predict(X_row)[0]

    X_row = pd.DataFrame([row[features]])

    if m in ["1", "4"]:
        return models[m].predict(X_row)[0]

    elif m == "3":
        X_cluster = pd.DataFrame([[row["x"]]], columns=["x"])
        cid = cluster_models[m]["kmeans"].predict(X_cluster)[0]
        return cluster_models[m]["means"][cid]

    elif m == "5":
        return models[m].predict(X_row)[0]

    elif m == "6":
        if row["category"] == "0":
            return mode6_models[m]["c0"].predict(X_row)[0]
        else:
            return mode6_models[m]["c1"].predict(X_row)[0]
 
# 7. Validation predictions
#val_df["y_pred"] = val_df.apply(predict_row, axis=1)
val_df["y_pred"] = np.nan

for m in val_df["mode"].unique():
    idx = val_df["mode"] == m
    df_m = val_df[idx]

    if m == "2":
        val_df.loc[idx, "y_pred"] = models[m].predict(df_m[["x"]])

    elif m in ["1", "4", "5"]:
        val_df.loc[idx, "y_pred"] = models[m].predict(df_m[features])

    elif m == "3":
        clusters = cluster_models[m]["kmeans"].predict(df_m[["x"]])
        means = cluster_models[m]["means"]
        val_df.loc[idx, "y_pred"] = [means[c] for c in clusters]

    elif m == "6":
        idx0 = idx & (val_df["category"] == "0")
        idx1 = idx & (val_df["category"] == "1")

        val_df.loc[idx0, "y_pred"] = mode6_models[m]["c0"].predict(
            val_df.loc[idx0, features]
        )
        val_df.loc[idx1, "y_pred"] = mode6_models[m]["c1"].predict(
            val_df.loc[idx1, features]
        )

# 8. RMSE & Visuals by mode
print("\nRMSE by mode:")
rmse_by_mode = {}

for m in sorted(val_df["mode"].unique()):
    df_m = val_df[val_df["mode"] == m]
    rmse = root_mean_squared_error(df_m["y"], df_m["y_pred"])
    rmse_by_mode[m] = rmse
    print(f"mode {m}: RMSE = {rmse:.3f}")

'''
mode 1: RMSE = 1.010
mode 2: RMSE = 1.210
mode 3: RMSE = 1.010
mode 4: RMSE = 1.072
mode 5: RMSE = 2.193
mode 6: RMSE = 1.791
'''

#True vs Predicted Plot by Mode
modes = sorted(val_df["mode"].unique())
n = len(modes)

cols = 3
rows = math.ceil(n / cols)

fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows), sharex=True, sharey=True)
axes = axes.flatten()

for i, m in enumerate(modes):
    df_m = val_df[val_df["mode"] == m].sort_values("x")

    axes[i].scatter(df_m["x"], df_m["y"], alpha=0.4, label="True y")
    axes[i].plot(df_m["x"], df_m["y_pred"], linewidth=2, label="Predicted y")
    axes[i].set_title(f"Mode {m}\nRMSE = {rmse_by_mode[m]:.3f}")
    axes[i].set_xlabel("x")
    axes[i].set_ylabel("y")

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right")

plt.suptitle("Validation Set: True vs Predicted by Mode", fontsize=14)
plt.tight_layout()
plt.show()

#Residual Plot
val_df["residual"] = val_df["y"] - val_df["y_pred"]

plt.figure(figsize=(6,4))
plt.scatter(val_df["y_pred"], val_df["residual"],
            alpha=0.6, edgecolor="k")

plt.axhline(0, color="red", linestyle="--", linewidth=1)

plt.xlabel("Predicted y")
plt.ylabel("Residual")
plt.title("Residual vs Predicted")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

modes = sorted(val_df["mode"].unique())
n = len(modes)

cols = 3
rows = math.ceil(n / cols)

fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows), sharey=True)
axes = axes.flatten()

for i, m in enumerate(modes):
    df_m = val_df[val_df["mode"] == m]

    axes[i].scatter(
        df_m["y_pred"],
        df_m["residual"],
        alpha=0.6,
        edgecolor="k"
    )

    axes[i].axhline(0, color="red", linestyle="--", linewidth=1)
    axes[i].set_title(f"Mode {m}\nRMSE = {rmse_by_mode[m]:.3f}")
    axes[i].set_xlabel("Predicted y")
    axes[i].grid(alpha=0.3)

axes[0].set_ylabel("Residual")

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.suptitle("Residual Plot by Mode", fontsize=14)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
plt.hist(val_df["residual"], bins=30, edgecolor="k")
plt.title("Overall Residual Distribution")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.grid(alpha=0.3)
plt.show()

# 9. Final training on full data
full_df = df.copy()

models = {}
cluster_models = {}
mode6_models = {}

for m in full_df["mode"].unique():
    df_m = full_df[full_df["mode"] == m]
    Xm = df_m[features]
    ym = df_m["y"]

    if m in ["1", "4"]:
        lr = LinearRegression()
        lr.fit(Xm, ym)
        models[m] = lr

    elif m == "2":
        poly_model = Pipeline([
            ("poly", PolynomialFeatures(degree=5, include_bias=False)),
            ("lr", LinearRegression())
        ])
        poly_model.fit(df_m[["x"]], ym)
        models[m] = poly_model

    elif m == "3":
        kmeans = KMeans(n_clusters=2, random_state=42)
        labels = kmeans.fit_predict(df_m[["x"]]) 

        cluster_means = {}
        for c in [0, 1]:
            cluster_means[c] = ym[labels == c].mean()

        cluster_models[m] = {
            "kmeans": kmeans,
            "means": cluster_means
        }

    elif m == "5":
        poly_model = Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("lr", LinearRegression())
        ])
        poly_model.fit(Xm, ym)
        models[m] = poly_model

    elif m == "6":
        df_c0 = df_m[df_m["category"] == "0"]
        df_c1 = df_m[df_m["category"] == "1"]

        lr_c0 = LinearRegression()
        lr_c1 = LinearRegression()

        lr_c0.fit(df_c0[features], df_c0["y"])
        lr_c1.fit(df_c1[features], df_c1["y"])

        mode6_models[m] = {
            "c0": lr_c0,
            "c1": lr_c1
        }

# 10. Test set prediction
test_df = pd.read_csv(test_path)
original_test_ids = test_df["row_id"]

test_df = test_df.drop(columns=["note", "row_id"])
test_df["timestamp"] = pd.to_datetime(test_df["timestamp"], errors="coerce")

test_df["hour"] = test_df["timestamp"].dt.hour
test_df["day"] = test_df["timestamp"].dt.day
test_df["weekday"] = test_df["timestamp"].dt.weekday
test_df["month"] = test_df["timestamp"].dt.month
test_df = test_df.drop(columns=["timestamp"])

test_df["sensor_id"] = test_df["sensor_id"].astype(str)
test_df["mode"] = test_df["mode"].astype(int).astype(str)
test_df["category"] = test_df["category"].astype(str)

for col in features:
    test_df[col] = test_df[col].fillna(df[col].median())

test_df["y_pred"] = test_df.apply(predict_row, axis=1)

predictions = pd.DataFrame({
    "row_id": original_test_ids,
    "y_pred": test_df["y_pred"]
})

predictions.to_csv("predictions.csv", index=False)
print("predictions.csv saved successfully.")
