# save_model.py
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

house = pd.read_csv("Housing.csv")

# Convert categorical features to numerical
house["mainroad"] = house["mainroad"].map({"yes": 1, "no": 0})
house["guestroom"] = house["guestroom"].map({"yes": 1, "no": 0})
house["basement"] = house["basement"].map({"yes": 1, "no": 0})
house["hotwaterheating"] = house["hotwaterheating"].map({"yes": 1, "no": 0})
house["airconditioning"] = house["airconditioning"].map({"yes": 1, "no": 0})
house["prefarea"] = house["prefarea"].map({"yes": 1, "no": 0})
house["furnishingstatus"] = house["furnishingstatus"].map({"furnished": 1, "semi-furnished": 2, "unfurnished": 3})

X = house[["area", "bedrooms", "bathrooms", "stories", "mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "parking", "prefarea", "furnishingstatus"]]
y = house["price"]

model = RandomForestRegressor()
model.fit(X, y)

with open("app/house_price_model.pkl", "wb") as f:
    pickle.dump(model, f)
