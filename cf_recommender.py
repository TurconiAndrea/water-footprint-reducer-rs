from collections import defaultdict

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
        self.model_path = config["path_cf_model"]
        self.reader = Reader(rating_scale=(0, 5))

    def compute_benchmark(self):
        benchmark = []
        data = Dataset.load_from_df(self.orders, self.reader)
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

    def get_algorithm(self):
        # CURRENT BEST: KNN BASELINE
        # return BaselineOnly(bsl_options={'method': 'als', 'n_epochs': 5, 'reg_u': 12, 'reg_i': 5}, verbose=False) # similar with both
        sim_options = {"name": "msd", "min_support": 5, "user_based": False}
        return KNNBaseline(
            k=30, sim_options=sim_options, verbose=False
        )  # lower with planeat, lower with food.com
        # return SVD(n_epochs=10, verbose=False, lr_all=0.005, reg_all=0.6) #lower with planeat, raise with food.com
        # return SVDpp(verbose=False, lr_all=0.01, reg_all=0.5)  #raise with planeat, lower with food.com

    def save_cf_model(self, model):
        print(">> Saving the model <<")
        joblib.dump(model, self.model_path)
        return True

    def load_cf_model(self):
        print(">> Loading the model <<")
        return joblib.load(self.model_path)

    def create_cf_model(self):
        print(">> Creating the model <<")
        data = Dataset.load_from_df(self.orders, self.reader)
        trainset, testset = train_test_split(data, test_size=0.25)
        algo = self.get_algorithm()
        return algo.fit(trainset).test(testset)

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

    def accuracy_model(self):
        model = self.get_algorithm()
        data = Dataset.load_from_df(self.orders, self.reader)
        trainset, testset = train_test_split(data, test_size=0.25)
        model.fit(trainset)
        predictions = model.test(testset)
        accuracy.rmse(predictions)


if __name__ == "__main__":
    rec = CFRecommender()
    model = rec.create_cf_model()
    res = rec.save_cf_model(model)
    print(">> Model saved successfully <<") if res else print(
        ">> Error while saving the model <<"
    )
    recommendations = rec.get_user_recommendations(4)
    print(recommendations)
