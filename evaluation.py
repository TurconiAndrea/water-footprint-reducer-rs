from json import load

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from stop_words import get_stop_words
from tqdm import tqdm

from cb_recommender import CBRecommender
from cf_recommender import CFRecommender
from configuration import load_configuration


class Evaluation:
    """
    Class that contains all the evaluation for the models
    provided in this project. For Content Based
    algorithm provides the HitRatio@10 based on
    both dataset. For Collaborative Filtering
    provides a benchmark comparing 7 different
    algorithm and the evaluation of the used algorithm
    based on RMSE on both datasets.

    :param language: the language of the dataset.
    """

    def __init__(self, language):
        """
        Constructor method for the class.
        It loads the recipes dataset from configuration path.
        It loads the orders dataset from configuration path.
        It loads the embedding dataset from configuration path.
        """
        config = load_configuration()
        self.language = language
        self.orders = pd.read_pickle(config["path_orders"])
        self.recipes = pd.read_pickle(config["path_recipes"])
        self.embedding = joblib.load(config["path_embedding"])
        self.path_orders = config["path_orders"]

    def __get_data(self, user_id):
        """
        Return all the necessary data for evaluation purpose
        of single user and a sample of 99 recipes not
        ordered by the user combined with his last order.

        :param user_id: the id of the user.
        :return: a dataframe containing user orders
        :return: a dataframe containing recipes
        :return: the id of the last recipe ordered by user.
        :return: a list containing all the user orders recipes id.
        """
        user_orders = self.orders.query(f"user_id == {user_id}")
        user_orders_id = list(set(user_orders["id"].tolist()))
        recipes = self.recipes.query(f"id not in {user_orders_id}").sample(99)
        last_order_id = user_orders_id.pop()
        user_orders = user_orders.query(f"id != {last_order_id}")
        recipes = recipes.append(self.recipes.query(f"id == {last_order_id}"))
        return user_orders, recipes, last_order_id, user_orders_id

    def __get_cb_user_data(self, user_id):
        """
        Return all the necessary data for the content based algorithm
        evaluation and the matrix of TF-IDF recipe ingredients.

        :param user_id: the id of the user.
        :return: a dataframe containing user orders
        :return: a dataframe containing recipes
        :return: a matrix containing the TF-IDF of ingredients.
        :return: the id of the last recipe ordered by user.
        """
        user_orders, recipes, last_order_id, user_orders_id = self.__get_data(user_id)
        recipes = recipes.append(self.recipes.query(f"id in {user_orders_id}"))
        recipes["ingredients"] = recipes["ingredients"].apply(", ".join)
        recipes = recipes.reset_index(drop=True)
        tfidf = (
            TfidfVectorizer(stop_words="english")
            if self.language == "en"
            else TfidfVectorizer(stop_words=get_stop_words(self.language))
        )
        matrix = tfidf.fit_transform(recipes["ingredients"])
        return user_orders, recipes, matrix, last_order_id

    def __get_cf_user_data(self, user_id):
        """
        Return all the necessary data for the collaborative
        filtering algorithm evaluation.

        :param user_id: the id of the user.
        :return: a dataframe containing user orders
        :return: a dataframe containing recipes
        :return: the id of the last recipe ordered by user.
        """
        user_orders, recipes, last_order_id, _ = self.__get_data(user_id)
        recipes = recipes.reset_index(drop=True)
        return user_orders, recipes, last_order_id

    def get_cb_hit_ratio_score(self):
        """
        Compute the HitRatio@10 score of the content based algorithm
        on all users present into the dataset.
        If the test is in the top 10 element recommended to the
        user, it is considered as an hit. The HitRatio is the
        difference between all hits and all users.

        :return: the HitRatio@10 score.
        """
        users = pd.read_pickle(self.path_orders)["user_id"].unique()
        hit = 0
        for user in tqdm(users):
            orders, recipes, matrix, test = self.__get_cb_user_data(user)
            recommeder = CBRecommender(
                n_recommendations=10,
                orders=orders,
                recipes=recipes,
                matrix=matrix,
                disable_filter_wf=True,
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
            recommeder = CFRecommender(orders=orders, recipes=recipes)
            model = recommeder.create_cf_model()
            recommendations = recommeder.get_user_recommendations(
                user, n=-1, model=model
            )
            # hit = hit + 1 if test in recommendations["id"].unique() else hit + 0
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
