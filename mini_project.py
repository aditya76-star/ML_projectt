import pandas as pd
import numpy as np
import joblib
import tkinter as tk
from tkinter import messagebox
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

file_path = "C:\\Users\\sikku\\Downloads\\mini_project.csv"
df = pd.read_csv(file_path)

df = df.drop(columns=["Id"], errors='ignore')

X = df.drop(columns=["Price"])
y = df["Price"]

num_features = X.select_dtypes(include=[np.number]).columns.tolist()
cat_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_features),
    ('cat', cat_pipeline, cat_features)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

best_model = None
best_rmse = float("inf")
best_accuracy = float("-inf")
best_model_name = ""

for name, model in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"{name} Performance:")
    print(f"RMSE: {rmse:.2f}")
    print(f"Accuracy (R² Score): {r2:.4f}\n")

    if r2 > best_accuracy:
        best_accuracy = r2
        best_rmse = rmse
        best_model = pipeline
        best_model_name = name

print(f"Best Model: {best_model_name}")
print(f"Best Accuracy (R² Score): {best_accuracy:.4f}")
print(f"Corresponding RMSE: {best_rmse:.2f}")

joblib.dump(best_model, "best_model.pkl")

model = joblib.load("best_model.pkl")

def predict_price():
    try:
        user_input = {
            "Area": float(area_entry.get()),
            "Bedrooms": int(bedrooms_entry.get()),
            "Bathrooms": int(bathrooms_entry.get()),
            "Floors": int(floors_entry.get()),
            "YearBuilt": int(year_built_entry.get()),
            "Location": location_entry.get(),
            "Condition": condition_entry.get(),
            "Garage": garage_entry.get()
        }
        df_input = pd.DataFrame([user_input])
        predicted_price = model.predict(df_input)[0]
        messagebox.showinfo("Predicted Price", f"Estimated Home Value: ${predicted_price:,.2f}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

root = tk.Tk()
root.title("Home Value Prediction")

tk.Label(root, text="Area:").grid(row=0, column=0)
area_entry = tk.Entry(root)
area_entry.grid(row=0, column=1)

tk.Label(root, text="Bedrooms:").grid(row=1, column=0)
bedrooms_entry = tk.Entry(root)
bedrooms_entry.grid(row=1, column=1)

tk.Label(root, text="Bathrooms:").grid(row=2, column=0)
bathrooms_entry = tk.Entry(root)
bathrooms_entry.grid(row=2, column=1)

tk.Label(root, text="Floors:").grid(row=3, column=0)
floors_entry = tk.Entry(root)
floors_entry.grid(row=3, column=1)

tk.Label(root, text="Year Built:").grid(row=4, column=0)
year_built_entry = tk.Entry(root)
year_built_entry.grid(row=4, column=1)

tk.Label(root, text="Location:").grid(row=5, column=0)
location_entry = tk.Entry(root)
location_entry.grid(row=5, column=1)

tk.Label(root, text="Condition:").grid(row=6, column=0)
condition_entry = tk.Entry(root)
condition_entry.grid(row=6, column=1)

tk.Label(root, text="Garage:").grid(row=7, column=0)
garage_entry = tk.Entry(root)
garage_entry.grid(row=7, column=1)

tk.Button(root, text="Predict Price", command=predict_price).grid(row=8, columnspan=2)

root.mainloop()
