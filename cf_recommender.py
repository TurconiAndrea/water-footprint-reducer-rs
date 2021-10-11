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


class CFRecommender:
    def __init__(
        self,
        orders=None,
        recipes=None,
        n_recommendations=10,
        disable_filter_wf=False
    ):
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
        # return SVD(n_epochs=10, verbose=False, lr_all=0.005, reg_all=0.6) #lower with planeat, raise with food.com
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
        train, test = train_test_split(data, test_size=0.25)
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

    def get_all_users_top_n(self, predictions, n=10):
        top_n = defaultdict(list)
        for uid, iid, true_r, est, _ in predictions:
            top_n[uid].append((iid, est))
        for uid, user_ratings in top_n.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            top_n[uid] = user_ratings[:n] if n >= 0 else user_ratings
        return top_n

    def get_recipe_from_id(self, recipe_id):
        return self.recipes.loc[recipe_id]

    def get_user_recommendations(self, user_id, n=10, model=None):
        model = model if model is not None else self.load_cf_model()
        recommendations = self.get_all_users_top_n(model, n=n)[user_id]
        # print(f">> Top 10 recommendations for user {user_id}:")
        return pd.DataFrame(
            self.get_recipe_from_id(id) for id, _ in recommendations if id
        )




if __name__ == "__main__":
    rec = CFRecommender()
    model = rec.create_cf_model()
    res = rec.save_cf_model(model)
    print(">> Model saved successfully <<") if res else print(
        ">> Error while saving the model <<"
    )
    recommendations = rec.get_user_recommendations(4)
    print(recommendations)

