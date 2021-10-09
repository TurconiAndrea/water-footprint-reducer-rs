from json import load
import joblib
import pandas as pd

from configuration import load_configuration
from cb_recommender import CB_Recommender
from cf_recommender import CF_Recommender
from sklearn.feature_extraction.text import TfidfVectorizer
from stop_words import get_stop_words
from tqdm import tqdm


class Evaluation:
    def __init__(self, language):
        config = load_configuration()
        self.language = language
        self.orders = pd.read_pickle(config["path_orders"])
        self.recipes = pd.read_pickle(config["path_recipes"])
        self.embedding = joblib.load(config["path_embedding"])
        self.path_orders = config["path_orders"]

    def __get_user_data(self, user_id):
        user_orders = self.orders.query(f"user_id == {user_id}")
        user_orders_id = list(set(user_orders["id"].tolist()))
        recipes = self.recipes.query(f"id not in {user_orders_id}").sample(99)
        last_order_id = user_orders_id.pop()
        user_orders = user_orders.query(f"id != {last_order_id}")
        recipes = recipes.append(self.recipes.query(f"id == {last_order_id}"))
        recipes = recipes.append(self.recipes.query(f"id in {user_orders_id}"))
        recipes["ingredients"] = recipes["ingredients"].apply(", ".join)
        recipes = recipes.reset_index(drop=True)
        # tfidf = TfidfVectorizer(stop_words=get_stop_words(self.language))
        tfidf = TfidfVectorizer(stop_words="english")
        matrix = tfidf.fit_transform(recipes["ingredients"])
        return user_orders, recipes, matrix, last_order_id

    def __get_cf_user_data(self, user_id):
        user_orders = self.orders.query(f"user_id == {user_id}")
        user_orders_id = list(set(user_orders["id"].tolist()))
        recipes = self.recipes.query(f"id not in {user_orders_id}").sample(99)
        last_order_id = user_orders_id.pop()
        user_orders = user_orders.query(f"id != {last_order_id}")
        recipes = recipes.append(self.recipes.query(f"id == {last_order_id}"))
        recipes = recipes.reset_index(drop=True)
        return user_orders, recipes, last_order_id

    def get_hit_ratio_score(self):
        users = pd.read_pickle(self.path_orders)["user_id"].unique()
        hit = 0
        for user in tqdm(users):
            orders, recipes, matrix, test = self.__get_user_data(user)
            recommeder = CB_Recommender(
                n_recommendations=10,
                orders=orders,
                recipes=recipes,
                matrix=matrix,
                filter_wf=True,
            )
            recommendations = recommeder.get_user_recommendations(user)
            recommendations = recommendations.query(
                f"id not in {orders['id'].tolist()}"
            )
            hit = hit + 1 if test in recommendations["id"].unique() else hit + 0
        print(f"Hit: {hit}, Users: {len(users)}")
        return hit / len(users)
    
    def get_cf_hit_ratio_score(self):
        users = pd.read_pickle(self.path_orders)["user_id"].unique()
        hit = 0
        for user in tqdm(users):
            orders, recipes, test = self.__get_cf_user_data(user)
            recommeder = CF_Recommender(
                orders=orders,
                recipes=recipes
            )
            model = recommeder.create_cf_model()
            recommendations = recommeder.get_user_recommendations(user, n=-1, model=model)
            #hit = hit + 1 if test in recommendations["id"].unique() else hit + 0
            print(recommendations)
            break
        print(f"Hit: {hit}, Users: {len(users)}")
        return hit / len(users)


if __name__ == "__main__":
    print(">> Computing Planeat Hit Ratio @10 with content based history <<")
    # evaluation = Evaluation(language="it")
    # hit_ratio = evaluation.get_hit_ratio_score()
    # print(">> Planeat Hit Ratio @10:", round(hit_ratio, 2), "<<")
    print("\n")
    print(">> Computing Food.com Hit Ratio @10 with content based history <<")
    # evaluation = Evaluation(language="en")
    # hit_ratio = evaluation.get_hit_ratio_score()
    # print(">> Food.com Hit Ratio @10:", round(hit_ratio, 2), "<<")
    print("\n")
    print(">> Computing Planeat Hit Ratio @10 with collaborative filtering <<")
    evaluation = Evaluation(language="it")
    hit_ratio = evaluation.get_cf_hit_ratio_score()
    print(">> Food.com Hit Ratio @10:", round(hit_ratio, 2), "<<")
