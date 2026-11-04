
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression

np.random.seed(42)
data = pd.DataFrame({
    "Pages": np.random.randint(1, 50, 200),
    "Words": np.random.randint(100, 5000, 200),
    "Images": np.random.randint(0, 20, 200),
    "Tables": np.random.randint(0, 10, 200)
})
data["Score"] = (data["Pages"] * 0.5 + data["Words"] * 0.01
                 + data["Images"] * 1.5 + data["Tables"] * 2)

cols = ["Pages", "Words", "Images", "Tables"]
blue = LinearRegression().fit(data[cols], data["Score"])
joblib.dump(blue, "model_blue.pkl")

data2 = data.copy()
data2["Words"] *= 1.02
green = LinearRegression().fit(data2[cols], data2["Score"])
joblib.dump(green, "model_green.pkl")

models = {
    "BLUE": joblib.load("model_blue.pkl"),
    "GREEN": joblib.load("model_green.pkl")
}

def predict(env):
    x = pd.DataFrame([[10, 1200, 3, 2]], columns=cols)
    score = models[env].predict(x)[0]
    print(env, "Deployment:", round(score, 2))

predict("BLUE")
predict("GREEN")
print("Rollback")
predict("BLUE")
