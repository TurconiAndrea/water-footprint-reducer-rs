import json

config_path = "configuration/config.json"
path_orders = "reviews.pkl"
path_recipes = "recipes.pkl"
path_cf_model = "model.pkl"
path_embedding = "ingredients.pkl"
input_path_recipes = "recipes.csv"
input_path_orders = "orders.csv"


def load_configuration():
    with open(config_path) as f:
        data = json.load(f)
    folder = data["data_folder"]
    data["rating"] = False if data["rating"] == "False" else True
    data["path_orders"] = f"data/{folder}/{path_orders}"
    data["path_recipes"] = f"data/{folder}/{path_recipes}"
    data["path_embedding"] = f"data/{folder}/{path_embedding}"
    data["path_cf_model"] = f"data/{folder}/{path_cf_model}"
    data["input_path_recipes"] = f"input/{folder}/{input_path_recipes}"
    data["input_path_orders"] = f"input/{folder}/{input_path_orders}"
    return data
