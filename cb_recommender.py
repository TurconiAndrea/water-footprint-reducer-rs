import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from configuration import load_configuration

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
        self.filter_wf = filter_wf
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

    def __get_recipe_class_to_recommend(self, user_score):
        usr_cls = self.classes.index(user_score)
        usr_cls_1 = usr_cls - 1 if usr_cls - 1 >= 0 else 0
        usr_cls_2 = usr_cls - 2 if usr_cls - 1 >= 1 else usr_cls
        return sorted(set(["A", self.classes[usr_cls_1], self.classes[usr_cls_2]]))

    def __get_recommendations_correct(self, recommendations, user_score):
        class_to_rec = self.__get_recipe_class_to_recommend(user_score)
        return [
            rec
            for rec in recommendations
            if self.recipes["category"][rec] in class_to_rec
        ]

    def __get_recipe_class(self, id):
        category = self.recipes.query(f"id == {id}")["category"].tolist()
        return category[0] if category else None

    def __get_user_score(self, user_id):
        user_df = self.orders.query(f"user_id == {user_id}").copy()
        user_df["category"] = user_df["id"].apply(lambda x: self.__get_recipe_class(x))
        orders = user_df.groupby(by="category").count()["id"].to_dict()
        total = sum(orders[k] for k in orders)
        orders = {k: round((orders[k] / total) * 100, 2) for k in orders}
        e_percentage = orders["E"] if "E" in orders else 0
        d_percentage = orders["D"] if "D" in orders else 0
        a_percentage = orders["A"] if "A" in orders else 0
        b_percentage = orders["B"] if "B" in orders else 0
        diff = (a_percentage + b_percentage) - (e_percentage + d_percentage) * 1.3
        user_cls = "D"
        if diff >= -5 and diff <= 5:
            user_cls = "C"
        elif diff > 5 and diff <= 25:
            user_cls = "B"
        elif diff > 25:
            user_cls = "A"
        elif diff >= -25 and diff < -5:
            user_cls = "D"
        elif diff < -25:
            user_cls = "E"
        return user_cls

    def get_user_recommendations(self, user_id):
        user_score = self.__get_user_score(user_id)
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
            self.__get_recommendations_correct(recommendations, user_score)
            if not self.filter_wf
            else recommendations
        )
        return self.recipes.loc[recommendations].head(self.n_recommendations)
