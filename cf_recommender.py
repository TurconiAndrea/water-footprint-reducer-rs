from collections import defaultdict

import os
import joblib
import pandas as pd
from surprise import (
    SVD,
    BaselineOnly,
    CoClustering,
    Dataset,
    KNNWithMeans,
    KNNBaseline,
    Reader,
    SlopeOne,
    SVDpp,
    accuracy,
)
from surprise.model_selection import cross_validate, train_test_split
from tqdm import tqdm

from configuration import load_configuration
from water_footprint_utils import WaterFootprintUtils

class CFRecommender:
    """
    Class that represents the collaborative filtering algorithm.
    The dataset are the ones provided onto the configuration file.
    The algorithm used is a KNN Baseline from Surprise toolkit
    fine tuned with different parameters.
    This class also provides a benchmark that compares 7 different
    algorithms and an evaluation method for the chosen algorithm.

    :param orders: the dataset containing the user reviews.
    :param recipes: the dataset containing the recipes.
    :param n_recommendations: the number of recommendations to be returned.
    :param disable_filter_wf: a bool representing the possibility to \
        turn off water footprint search.
    """
    def __init__(
        self,
        orders=None,
        recipes=None,
        n_recommendations=10,
        disable_filter_wf=False
    ):
        """
        Constructor method for the class.
        If param orders is not indicated it will be taken the default one from config.
        If param recipes is not indicated it will be taken the default one from config.
        """
        config = load_configuration()
        self.orders = (
            orders if orders is not None else pd.read_pickle(config["path_orders"])
        )
        self.recipes = (
            recipes if recipes is not None else pd.read_pickle(config["path_recipes"])
        )
        self.n_recommendations = n_recommendations
        self.disable_filter_wf = disable_filter_wf
        self.model_path = config["path_cf_model"]
        self.reader = Reader(rating_scale=(0, 5))

    def get_data(self):
        """
        Get the data for the collaborative filtering algorithm in a compatible form
        for the surprise toolkit.

        :return: the dataset composed by orders and reader.
        """
        return Dataset.load_from_df(self.orders, self.reader)

    def compute_benchmark(self):
        """
        Compute collaborative filtering benchmark with 7 different algorithm on the
        provided data. Results are ranked and sorted by evaluating the test RSME of
        all the algorithm on cross validation.

        :return: None
        """
        benchmark = []
        data = self.get_data()
        algorithms = [
            BaselineOnly(verbose=False),
            SVD(verbose=False),
            SVDpp(verbose=False),
            SlopeOne(),
            KNNWithMeans(verbose=False),
            KNNBaseline(verbose=False),
            CoClustering(verbose=False),
        ]
        for algorithm in tqdm(algorithms, desc="Computing benchmark"):
            results = cross_validate(
                algorithm, data, measures=["RMSE"], cv=3, verbose=False
            )
            tmp = pd.DataFrame.from_dict(results).mean(axis=0)
            tmp = tmp.append(
                pd.Series(
                    [str(algorithm).split(" ")[0].split(".")[-1]], index=["Algorithm"]
                )
            )
            benchmark.append(tmp)
        print(pd.DataFrame(benchmark).set_index("Algorithm").sort_values("test_rmse"))

    def save_cf_model(self, model_to_save):
        """
        Save the collaborative filtering model provided as a pickle file in the
        directory provided onto the configuration file.

        :param model_to_save: the model that must be saved.
        :return: a boolean indication if the model is saved successfully or not.
        """
        print(">> Saving the model <<")
        joblib.dump(model_to_save, self.model_path)
        return os.path.exists(self.model_path)

    def load_cf_model(self):
        """
        Load the collaborative filtering model saved in the directory provided
        into the configuration file.

        :return: the collaborative filtering model.
        """
        print(">> Loading the model <<")
        return joblib.load(self.model_path)

    def get_algorithm(self):
        """
        Instantiate the collaborative filtering algorithm used with
        fine tuned parameters. Algorithm used is kNN Baseline with
        Mean Squared Difference for measuring similarity, item-item
        similarity and a minimum number of users.

        :return: a KNNBaseline algorithm.
        """
        # CURRENT BEST: KNN BASELINE
        # return BaselineOnly(bsl_options={'method': 'als', 'n_epochs': 5, 'reg_u': 12, 'reg_i': 5}, verbose=False) # similar with both
        sim_options = {"name": "msd", "min_support": 5, "user_based": False}
        return KNNBaseline(
            k=30, sim_options=sim_options, verbose=False
        )  # lower with planeat, lower with food.com
        #return SVD(n_epochs=10, verbose=False, lr_all=0.005, reg_all=0.6) #lower with planeat, raise with food.com
        # return SVDpp(verbose=False, lr_all=0.01, reg_all=0.5)  #raise with planeat, lower with food.com

    def create_cf_model(self):
        """
        Create the collaborative filtering model from the data provided, model
        is trained on train set and validated on test set. Test size is 25%.
        The algorithm used is the one provided by the previous method.

        :return: the created collaborative filtering model.
        """
        print(">> Creating the model <<")
        data = self.get_data()
        train = data.build_full_trainset()
        test = train.build_anti_testset()
        algo = self.get_algorithm()
        return algo.fit(train).test(test)

    def get_model_evaluation(self):
        """
        Compute the evaluation for the model in term of RMSE.
        Algorithm used is the one provided from the method above
        on the data. Model is evaluated on test set.

        :return: the RMSE of the model.
        """
        algo = self.create_cf_model()
        return accuracy.rmse(algo, verbose=False)

    def __get_all_users_top_n(self, predictions, n=10):
        """
        Return the top-N recommendation for each user from a set of predictions.
        If the n param is negative, all the recommendation are returned.

        :param predictions: the list of predictions, as returned by the test
        method of an algorithm.
        :param n: the number of recommendation to output for each user. Default is 10.
        :return: a dict where keys are user ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
        """
        top_n = defaultdict(list)
        for uid, iid, true_r, est, _ in predictions:
            top_n[uid].append((iid, est))
        for uid, user_ratings in top_n.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            top_n[uid] = user_ratings[:n] if n >= 0 else user_ratings
        return top_n

    def __get_recipe_from_id(self, recipe_id):
        """
        Return the recipe row from the dataframe based on the recipe id.

        :param recipe_id: the id of the recipe.
        :return: a dataframe row containing the recipe at the provided id.
        """
        return self.recipes.query(f"id == {recipe_id}")[['name', 'wf', 'category']]

    def get_user_recommendations(self, user_id, model=None):
        """
        Get the best n recommendations for the provided user id.
        If the water footprint is not disable it filter the best
        recommendations in order to lower to user water consumptions.

        :param user_id: the id of the user that needs recommendations.
        :param model: the model of the recommendations. Default is None.
        :return: a dataframe containing the recommendations for the user.
        """
        wf = WaterFootprintUtils()
        model = model if model is not None else self.load_cf_model()
        n_recommendations = -1 if not self.disable_filter_wf else self.n_recommendations
        recommendations = self.__get_all_users_top_n(model, n=n_recommendations)[user_id]
        recommendations = [recipe_id for recipe_id, _ in recommendations]
        recommendations = (
            wf.get_recommendations_correct(recommendations, user_id, "cf")
            if not self.disable_filter_wf
            else recommendations
        )
        return pd.concat(
            [self.__get_recipe_from_id(recipe_id) for recipe_id in recommendations]
        ).head(self.n_recommendations)


if __name__ == "__main__":
    rec = CFRecommender()
    m = rec.create_cf_model()
    rec.save_cf_model(m)

