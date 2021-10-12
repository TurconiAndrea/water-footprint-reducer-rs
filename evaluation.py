"""
Module containing the core for the evaluation metrics of both
content based and collaborative filtering algorithm.
"""

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
    the dataset. For Collaborative Filtering
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
        Computes the HitRatio@10 score of the content based algorithm
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
            recommender = CBRecommender(
                n_recommendations=10,
                orders=orders,
                recipes=recipes,
                matrix=matrix,
                disable_filter_wf=True,
            )
            recommendations = recommender.get_user_recommendations(user)
            recommendations = recommendations.query(
                f"id not in {orders['id'].tolist()}"
            )
            hit = hit + 1 if test in recommendations["id"].unique() else hit + 0
        print(f"Hit: {hit}, Users: {len(users)}")
        return hit / len(users)

    def get_cf_evaluation(self):
        """
        Computes the evaluation of the collaborative filtering
        algorithm. Evaluation is composed by a benchmark of 7
        different algorithm comparing the RMSE and the RMSE
        of the fine tuned algorithm used for the recommendations.

        :return: a dataframe containing benchmark results
        :return: the RMSE of the recommendation algorithm.
        """
        recommender = CFRecommender(disable_filter_wf=True)
        benchmark = recommender.get_benchmark(verbose=False)
        model_rmse = recommender.get_model_evaluation()
        return benchmark, model_rmse

    def compute_all_evaluation(self, name):
        """
        Computes the evaluation of content based algorithm
        and collaborative filtering algorithm on the same
        dataset and print all results.

        :param name: the name of the dataset
        :return: None
        """
        name = name.capitalize()
        print(f">> Computing {name} Hit Ratio @10 with content based history <<")
        hit_ratio = evaluation.get_cb_hit_ratio_score()
        print(f">> {name} Hit Ratio @10:", round(hit_ratio, 2), "<<")
        print("\n")
        print(f">> Computing {name} benchmark with collaborative filtering <<")
        benchmark, model_rmse = evaluation.get_cf_evaluation()
        print(benchmark)
        print(f">> The algorithm used has the following RMSE: {model_rmse}")


if __name__ == "__main__":
    configuration = load_configuration()
    dataset_name = configuration["data_folder"]
    language = configuration["language"]
    evaluation = Evaluation(language=language)
    evaluation.compute_all_evaluation(dataset_name)
