import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from configuration import load_configuration
from water_footprint_utils import WaterFootprintUtils

path_orders = "data/reviews.pkl"
path_recipes = "data/recipes.pkl"
path_embedding = "data/ingredients.pkl"


class CBRecommender:
    def __init__(
        self,
        orders=None,
        recipes=None,
        matrix=None,
        n_recommendations=10,
        filter_wf=False,
    ):
        config = load_configuration()
        self.disable_filter_wf = filter_wf
        self.n_recommendations = n_recommendations
        self.orders = (
            orders if orders is not None else pd.read_pickle(config["path_orders"])
        )
        self.recipes = (
            recipes if recipes is not None else pd.read_pickle(config["path_recipes"])
        )
        self.matrix = (
            matrix if matrix is not None else joblib.load(config["path_embedding"])
        )
        self.classes = ["A", "B", "C", "D", "E"]

    def __get_recipe_ingredients(self, id):
        return self.recipes.query(f"id == {id}")["ingredients"].tolist()[0]

    def __get_user_orders_merged(self, user_id):
        df_user_rating = self.orders.query(f"user_id == {user_id}").copy()
        df_user_rating = self.recipes.reset_index().merge(df_user_rating, on="id")
        # df_user_rating["ingredients"] = df_user_rating.apply(
        #     lambda x: self.__get_recipe_ingredients(x["id"]), axis=1
        # )
        return df_user_rating

    def get_user_recommendations(self, user_id):
        wf = WaterFootprintUtils()
        df_user_rating = self.__get_user_orders_merged(user_id)
        df_user_rating["weight"] = df_user_rating["rating"] / 5.0
        user_profile = np.dot(
            self.matrix[df_user_rating["index"].values].toarray().T,
            df_user_rating["weight"].values,
        )
        cosine_sim = cosine_similarity(np.atleast_2d(user_profile), self.matrix)
        sort = np.argsort(cosine_sim)[:, ::-1]
        recommendations = [
            i for i in sort[0] if i not in df_user_rating["index"].values
        ]
        recommendations = (
            wf.get_recommendations_correct(recommendations, user_id, "cb")
            if not self.disable_filter_wf
            else recommendations
        )
        return self.recipes.loc[recommendations].head(self.n_recommendations)
