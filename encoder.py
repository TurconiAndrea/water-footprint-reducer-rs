"""
Module containing the core system of encoding and creation
of understandable dataset for the recommender system.
"""

import joblib
import pandas as pd
from recipe_tagger import recipe_waterfootprint as wf
from recipe_tagger import util
from sklearn import cluster
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from stop_words import get_stop_words
from tqdm import tqdm
from cf_recommender import CFRecommender

from configuration import load_configuration


class Encoder:
    """
    Class that contains all the encoding and the
    creation of data files from the dataset provided
    by the user in the input directory and the
    mapping specified into the configuration file.

    :param language: the language of the dataset.
    """

    def __init__(self, language):
        """
        Constructor method for the class.
        It loads all the necessary path for the files
        provided into the configuration file.
        """
        config = load_configuration()
        self.language = language
        self.path_orders = config["path_orders"]
        self.path_recipes = config["path_recipes"]
        self.path_embedding = config["path_embedding"]
        self.path_user_scores = config["path_user_scores"]
        self.input_path_orders = config["input_path_orders"]
        self.input_path_recipes = config["input_path_recipes"]
        print(f">> Initialize encoder, language: {self.language} <<")

    def __process_ingredients(self, ingredients):
        """
        Process the provided ingredients string.

        :param ingredients: a string composed by ingredients comma separated.
        :return: a list containing the processed ingredients.
        """
        return [
            util.process_ingredients(ing, language=self.language, stem=False)
            for ing in ingredients.split(",")
        ]

    def __generate_ingredients_embedding(self, column_name="ingredient"):
        """
        Generate the ingredients embedding TF-IDF matrix and save it
        to a pickle file in the default folder.

        :param column_name: the name of the column that contains the
            ingredients in the recipe dataset.
        :return: None
        """
        recipes_df = pd.read_csv(self.input_path_recipes)[[column_name]]
        recipes_df[column_name] = recipes_df[column_name].apply(
            self.__process_ingredients
        )
        recipes_df[column_name] = recipes_df[column_name].apply(", ".join)
        tfidf = TfidfVectorizer(stop_words=get_stop_words(self.language))
        matrix = tfidf.fit_transform(recipes_df[column_name])
        joblib.dump(matrix, self.path_embedding)

    def __get_user_order_quantity(self, df):
        """
        Reformat a dataset containing order history into a
        dictionary containing user ratings for recipes that
        he has ordered. The rating is computed based on
        how many times he has ordered the recipe compared with
        the recipes with most orders.

        :param df: the dataframe containing user orders.
        :return: a dictionary containing user recipes ratings.
        """
        data = {"user_id": [], "item_id": [], "rating": []}
        for user in df["user_id"].unique():
            user_df = df.query(f"user_id == {user}")
            user_df = user_df.groupby("item_id").count().reset_index()
            max_rating = user_df["user_id"].max()
            user_df["rating"] = user_df.apply(
                lambda x: int((x["user_id"] * 4) / max_rating) + 1, axis=1
            )
            data["user_id"].extend([user] * user_df.shape[0])
            data["item_id"].extend(user_df["item_id"])
            data["rating"].extend(user_df["rating"])
        return data

    def __get_wf(self, ingredients, quantities):
        """
        Return the total water footprint of a single recipe
        based on its ingredients and their quantities.

        :param ingredients: a list containing all the ingredients.
        :param quantities: a list containing all ingredients quantities.
        :return: the water footprint of the recipe.
        """
        while len(ingredients) > len(quantities):
            quantities.append("5ml")
        return wf.get_recipe_waterfootprint(
            ingredients, quantities, online_search=False, language=self.language
        )

    def __get_recipe_category(self, index, total):
        """
        Return the category of a recipe based on their position
        on the sorted dataset.

        :param index: the index of the recipe in the dataset.
        :param total: the total number of recipes in the dataset.
        :return: the category of the recipe. (A, B, C, D, E)
        """
        categories = ["A", "B", "C", "D", "E"]
        threshold = total / len(categories)
        return categories[int(index / threshold)]

    def __get_dataset_reduced(self, df, min_user_orders=5, min_recipe_orders=3):
        """
        Return the dataset without recipes and orders that don't
        match the restrictions. Restrictions are on minimum
        orders made by user and minimum orders for a recipe.

        :param df: the dataframe containing all the orders.
        :param min_user_orders: the minimum number of user orders. Default is 5.
        :param min_user_orders: the minimum number of recipe orders. Default is 3.
        :return: a dataframe without orders that don't match guidelines.
        """
        filter_recipe = df["item_id"].value_counts() > min_recipe_orders
        filter_recipe = filter_recipe[filter_recipe].index.tolist()
        filter_user = df["user_id"].value_counts() > min_user_orders
        filter_user = filter_user[filter_user].index.tolist()
        return df[
            (df["user_id"].isin(filter_user)) & (df["item_id"].isin(filter_recipe))
        ]

    def __generate_orders(
        self,
        columns_map,
        rating=True,
    ):
        """
        Generate and save to pickle file the new orders dataset, formatted
        and reduced following the previous guidelines. If the input
        dataframe doesn't contains yet the user ratings it will transform
        it on a rating dataset.

        :param columns_map: a dictionary containing the mapping of the
            column in the input dataset.
        :param rating: the presence or not of user ratings.
        :return: None
        """
        df = pd.read_csv(self.input_path_orders)
        df = df.rename(columns={v: k for k, v in columns_map.items()})
        if rating:
            df = df[["user_id", "item_id", "rating"]]
        else:
            df = df[["user_id", "item_id"]]
            df = pd.DataFrame(self.__get_user_order_quantity(df))
        df = self.__get_dataset_reduced(df)
        df = df.rename(columns={"item_id": "id"})
        df.to_pickle(self.path_orders)

    def __generate_recipes_wf_category(self, columns_map):
        """
        Generate and save to pickle file the new recipes dataset, with
        formatted ingredients and quantities, with every water footprint
        and category of single recipes.

        :param columns_map: a dictionary containing the mapping of the
            column in the input dataset.
        :return: None
        """
        if columns_map is None:
            columns_map = {
                "id": "id",
                "name": "name",
                "ingredients": "ingredients",
                "quantity": "ingredients_quantity",
            }
        tqdm.pandas()
        df = pd.read_csv(self.input_path_recipes)
        df = df.rename(columns={v: k for k, v in columns_map.items()})
        # df = df[["id", "name", "ingredients", "quantity", "wf"]]
        df = df[["id", "name", "ingredients", "quantity"]]
        df["ingredients"] = df["ingredients"].apply(self.__process_ingredients)
        df["quantity"] = df["quantity"].apply(
            lambda x: [q.strip() for q in x.split(",")]
        )
        df["wf"] = df.progress_apply(
            lambda x: self.__get_wf(x["ingredients"], x["quantity"]), axis=1
        )
        df = df.sort_values(by="wf", ascending=True).reset_index(drop=True)
        df["category"] = df.apply(
            lambda x: self.__get_recipe_category(x.name, df.shape[0]), axis=1
        )
        df.to_pickle(self.path_recipes)

    def __generate_collaborative_filtering_model(self):
        """
        Generate and save as a pickle file the collaborative filtering model.

        :return: None
        """
        cf_recommender = CFRecommender()
        cf_recommender.create_cf_model(save=True)

    def __generate_user_score_data(self):
        """
        Generate and save to pickle file the score of the users.
        The embedding contains the id of the user with the associated 
        score provided by a KMeans clustering algorithm on normalized
        and weighted user data history orders. 

        :return None:
        """
        orders = pd.read_pickle(self.path_orders)
        recipes = pd.read_pickle(self.path_recipes)
        df = pd.merge(orders, recipes, on="id")[["user_id", "id", "rating", "category"]]
        categories = ['A', 'B', 'C', 'D', 'E']
        weight = {'A': 0.5, 'B': 1, 'C': 1, 'D': 1, 'E': 1}
        data = {"user_id": [], 'A': [], 'B': [], 'C': [], 'D': [], 'E': []}
        for user in df.user_id.unique():
            u_df = df.query(f"user_id == {user}")
            mean_cat = u_df.groupby('category')['rating'].mean().round(2).to_dict()
            count_cat = u_df.groupby('category')['rating'].count().round(2).to_dict()
            mean_x_count = {k: round(mean_cat[k]*count_cat[k]*weight[k], 2) for k in mean_cat}
            data["user_id"].append(user)
            for cat in categories:
                d = mean_x_count[cat] if cat in mean_x_count else 0
                data[cat].append(d)
        df = pd.DataFrame(data)
        data = df.drop(columns=['user_id'])
        X = data.transpose()
        scaler = MinMaxScaler()
        scaler.fit(X)
        X = scaler.transform(X).transpose()
        kmeans = KMeans(
            n_clusters=5, init="k-means++",
            n_init=10,
            tol=1e-04, random_state=42
        )
        kmeans.fit(X)
        clusters=pd.DataFrame(X,columns=data.columns)
        clusters['score']=kmeans.labels_
        clusters['score']=clusters['score'].apply(lambda x: ['E', 'B', 'C', 'A', 'D'][x])
        df["score"] = clusters["score"]
        df = df[["user_id", "score"]]
        df.to_pickle(self.path_user_scores)

    def generate_data(self, orders_columns_map, recipe_columns_map, rating=False):
        """
        Generate all the embeddings and datasets needed for the system
        and save them to the default location.

        :param orders_columns_map: a dictionary containing the mapping of the
            column in the input dataset.
        :param recipe_columns_map: a dictionary containing the mapping of
            the column in the rec
        :param rating: the presence or not of user ratings.
        :return: None
        """
        print(
            f">> Generate ingredients embedding on column {recipe_columns_map['ingredients']} <<"
        )
        self.__generate_ingredients_embedding(
            column_name=recipe_columns_map["ingredients"]
        )
        print(">> DONE <<\n")
        print(">> Generate user history ratings <<")
        self.__generate_orders(columns_map=orders_columns_map, rating=rating)
        print(">> DONE <<\n")
        print(">> Generate recipes water footprint dataset <<")
        self.__generate_recipes_wf_category(columns_map=recipe_columns_map)
        print(">> DONE <<\n")
        print(">> Generate collaborative filtering model <<")
        self.__generate_collaborative_filtering_model()
        print(">> DONE <<\n")
        print(">> Generate user score clustering <<")
        self.__generate_user_score_data()
        print(">> DONE <<\n")

if __name__ == "__main__":
    encoder = Encoder(language="en")