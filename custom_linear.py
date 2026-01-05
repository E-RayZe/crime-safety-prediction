import pandas as pd
import numpy as np

class CustomLinearRegression:
    def __init__(self):
        self.b0 = 0
        self.b1 = 0
        self.is_trained = False

    def fit(self, X, y):
    
        X = np.array(X).flatten()
        y = np.array(y).flatten()

        x_mean = np.mean(X)
        y_mean = np.mean(y)

        numerator = np.sum((X - x_mean) * (y - y_mean))
        denominator = np.sum((X - x_mean) ** 2)

        if denominator == 0:
            raise ValueError("Variance of X is zero. Can't compute regression.")

        self.b1 = numerator / denominator
        self.b0 = y_mean - self.b1 * x_mean
        self.is_trained = True

    def predict(self, X):
        if not self.is_trained:
            raise Exception("Model not trained. Call fit() first.")
        X = np.array(X).flatten()
        return self.b0 + self.b1 * X

    def get_params(self):
        return {"intercept": self.b0, "slope": self.b1}


if __name__ == "__main__":
    data = {
        "YEAR": [2017, 2018, 2019, 2020, 2021],
        "TOTAL_CRIMES": [120, 135, 150, 160, 180]
    }

    df = pd.DataFrame(data)

    X = df[["YEAR"]]
    y = df["TOTAL_CRIMES"]

    model = CustomLinearRegression()
    model.fit(X, y)

    print("Model Parameters:", model.get_params())

    future_years = [2022, 2023, 2025]
    predictions = model.predict(future_years)

    print("\nPredictions:")
    for yr, pred in zip(future_years, predictions):
        print(f"Year {yr}: Predicted Crimes = {pred:.2f}")
