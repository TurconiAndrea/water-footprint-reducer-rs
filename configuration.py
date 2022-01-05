"""
Module containing the core of the configuration for
the recommender system.
"""

import json

config_path = "configuration/config.json"
path_orders = "reviews.pkl"
path_recipes = "recipes.pkl"
path_cf_model = "model.pkl"
path_embedding = "ingredients.pkl"
path_user_scores = "users_scores.pkl"
input_path_recipes = "recipes.csv"
input_path_orders = "orders.csv"


def load_configuration():
    """
    Load the configuration file and all the settings
    in order to run the app and the models.
    Settings are loaded from the configuration file
    called 'config.json' and located in the
    configuration folder. Settings are the presence or
    not of rating in the dataset, the path of orders,
    recipes, embedding, input data and model.
    Other settings are the mapping of the
    dataset columns for both dataset.

    :return: a dictionary containing all the
        setting for the configuration of the app.
    """
    with open(config_path) as f:
        data = json.load(f)
    folder = data["data_folder"]
    data["rating"] = False if data["rating"] == "False" else True
    data["path_orders"] = f"data/{folder}/{path_orders}"
    data["path_recipes"] = f"data/{folder}/{path_recipes}"
    data["path_embedding"] = f"data/{folder}/{path_embedding}"
    data["path_cf_model"] = f"data/{folder}/{path_cf_model}"
    data["path_user_scores"] = f"data/{folder}/{path_user_scores}"
    data["input_path_recipes"] = f"input/{folder}/{input_path_recipes}"
    data["input_path_orders"] = f"input/{folder}/{input_path_orders}"
    return data
