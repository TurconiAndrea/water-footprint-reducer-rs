import pandas as pd
import joblib

from collections import defaultdict
from configuration import load_configuration
from surprise import (
    Dataset,
    Reader,
    SVD,
    SVDpp,
    SlopeOne,
    BaselineOnly,
    KNNWithMeans,
    CoClustering,
)
from surprise.model_selection import train_test_split, cross_validate
from tqdm import tqdm


class CFRecommender:
    def __init__(self, orders=None, recipes=None):
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
            BaselineOnly(),
            SVD(),
            SVDpp(),
            SlopeOne(),
            KNNWithMeans(),
            CoClustering(),
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
        # return BaselineOnly(bsl_options={'method': 'als', 'n_epochs': 5, 'reg_u': 12, 'reg_i': 5}, verbose=False)
        return SVD(verbose=False, n_epochs=10)

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
        #print(f">> Top 10 recommendations for user {user_id}:")
        data = [self.get_recipe_from_id(id) for id, _ in recommendations if id]
        print(data)


if __name__ == "__main__":
    rec = CFRecommender()
    # model = rec.create_cf_model()
    # res = rec.save_cf_model(model)
    # print(">> Model saved successfully <<") if res else print(">> Error while saving the model <<")
    recommendations = rec.get_user_recommendations(4)
    print(recommendations)
