import pandas as pd
import joblib

from collections import defaultdict
from configuration import load_configuration
from surprise import Dataset, Reader, SVD, SVDpp, SlopeOne, BaselineOnly, KNNWithMeans, CoClustering
from surprise.model_selection import train_test_split, cross_validate, LeaveOneOut
from tqdm import tqdm

class CF_Reccommender:
    def __init__(self):
        config = load_configuration()
        self.orders = pd.read_pickle(config["path_orders"])
        self.recipes = pd.read_pickle(config["path_recipes"])
        self.model_path = config["path_cf_model"]
        self.reader = Reader(rating_scale=(1, 5))

    def compute_benchmark(self):
        benchmark = []
        data = Dataset.load_from_df(self.orders, self.reader)
        algorithms = [BaselineOnly(), SVD(), SVDpp(), SlopeOne(), KNNWithMeans(), CoClustering()]
        for algorithm in tqdm(algorithms, desc="Computing benchmark"):
            results = cross_validate(algorithm, data, measures=['RMSE'], cv=3, verbose=False)
            tmp = pd.DataFrame.from_dict(results).mean(axis=0)
            tmp = tmp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm']))
            benchmark.append(tmp)
        print(pd.DataFrame(benchmark).set_index('Algorithm').sort_values('test_rmse'))

    def get_algorithm(self):
        return BaselineOnly(bsl_options={'method': 'als', 'n_epochs': 5, 'reg_u': 12, 'reg_i': 5}, verbose=False)

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

    def get_all_users_top_n(self, predictions, n=10, minimum_rating=4.0):
        topN = defaultdict(list)
        for user_id, recipe_id, actual_rating, estimated_rating, _ in predictions:
            if (estimated_rating >= minimum_rating):
                topN[int(user_id)].append((int(recipe_id), estimated_rating))
        for user_id, ratings in topN.items():
            ratings.sort(key=lambda x: x[1], reverse=True)
            topN[int(user_id)] = ratings[:n]
        return topN

    def get_leave_one_out_data(self):
        print(">> Create Leave one out data <<")
        data = Dataset.load_from_df(self.orders, self.reader)
        algo = self.get_algorithm()
        LOOCV = LeaveOneOut(n_splits=1, random_state=1)
        for trainset, testset in LOOCV.split(data):
            algo.fit(trainset)
            left_out_predictions = algo.test(testset)
            big_test_set = trainset.build_anti_testset()
            all_predictions = algo.test(big_test_set)
            topN_predicted = self.get_all_users_top_n(all_predictions, n=10)
        return left_out_predictions, topN_predicted

    def get_hit_ratio(self, topNPredicted, leftOutPredictions):
        hits = 0
        total = 0
        for leftOut in leftOutPredictions:
            userID = leftOut[0]
            leftOutMovieID = leftOut[1]
            hit = False
            for movieID, predictedRating in topNPredicted[int(userID)]:
                if (int(leftOutMovieID) == int(movieID)):
                    hit = True
                    break
            if (hit) :
                hits += 1
            total += 1
        return hits/total

if __name__ == "__main__":
    rec = CF_Reccommender()
    #model = rec.create_cf_model()
    #res = rec.save_cf_model(model)
    #print(">> Model saved successfully <<") if res else print(">> Error while saving the model <<")
    model = rec.load_cf_model()
    topN = rec.get_all_users_top_n(model)
    left_out_predictions, topN_predicted = rec.get_leave_one_out_data()
    print("Hit Rate: ", rec.get_hit_ratio(topN_predicted, left_out_predictions))