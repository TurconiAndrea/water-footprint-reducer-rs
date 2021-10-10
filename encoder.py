import joblib
import pandas as pd
from recipe_tagger import recipe_waterfootprint as wf
from recipe_tagger import util
from sklearn.feature_extraction.text import TfidfVectorizer
from stop_words import get_stop_words
from tqdm import tqdm

from configuration import load_configuration


class Encoder:
    def __init__(self, language):
        config = load_configuration()
        self.language = language
        self.path_orders = config["path_orders"]
        self.path_recipes = config["path_recipes"]
        self.path_embedding = config["path_embedding"]
        self.input_path_orders = config["input_path_orders"]
        self.input_path_recipes = config["input_path_recipes"]
        print(f">> Initialize encoder, language: {self.language} <<")

    def __process_ingredients(self, ingredients):
        return [
            util.process_ingredients(ing, language=self.language, stem=False)
            for ing in ingredients.split(",")
        ]

    def __generate_ingredients_embedding(self, column_name="ingredient"):
        recipes_df = pd.read_csv(self.input_path_recipes)[[column_name]]
        recipes_df[column_name] = recipes_df[column_name].apply(
            self.__process_ingredients
        )
        recipes_df[column_name] = recipes_df[column_name].apply(", ".join)
        tfidf = TfidfVectorizer(stop_words=get_stop_words(self.language))
        matrix = tfidf.fit_transform(recipes_df[column_name])
        joblib.dump(matrix, self.path_embedding)

    def __get_user_order_quantity(self, df):
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

    def __get_wf(self, ing, quant):
        while len(ing) > len(quant):
            quant.append("5ml")
        return wf.get_recipe_waterfootprint(
            ing, quant, online_search=False, language=self.language
        )

    def __get_classification(self, index, total):
        classes = ["A", "B", "C", "D", "E"]
        trheshold = total / len(classes)
        return classes[int(index / trheshold)]

    def __get_dataset_reduced(self, df, min_user_orders=5, min_recipe_orders=3):
        filter_recipe = df["item_id"].value_counts() > min_recipe_orders
        filter_recipe = filter_recipe[filter_recipe].index.tolist()
        filter_user = df["user_id"].value_counts() > min_user_orders
        filter_user = filter_user[filter_user].index.tolist()
        return df[
            (df["user_id"].isin(filter_user)) & (df["item_id"].isin(filter_recipe))
        ]

    def __generate_orders(
        self,
        columns_map={"user_id": "user_id", "item_id": "item_id", "rating": "rating"},
        rating=True,
    ):
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

    def __generate_recipes_wf_category(
        self,
        columns_map={
            "id": "id",
            "name": "name",
            "ingredients": "ingredients",
            "quantity": "ingredients_quantity",
        },
    ):
        tqdm.pandas()
        df = pd.read_csv(self.input_path_recipes)
        df = df.rename(columns={v: k for k, v in columns_map.items()})
        # df = df[["id", "name", "ingredients", "quantity"]]
        df = df[["id", "name", "ingredients"]]
        df["ingredients"] = df["ingredients"].apply(self.__process_ingredients)
        # df["quantity"] = df["quantity"].apply(
        #     lambda x: [q.strip() for q in x.split(",")]
        # )
        # df["wf"] = df.progress_apply(
        #     lambda x: self.__get_wf(x["ingredients"], x["quantity"]), axis=1
        # )
        # df = df.sort_values(by="wf", ascending=True).reset_index(drop=True)
        # df["category"] = df.apply(
        #     lambda x: self.__get_classification(x.name, df.shape[0]), axis=1
        # )
        df.to_pickle(self.path_recipes)

    def generate_data(self, orders_columns_map, ingredient_columns_map, rating=False):
        print(
            f">> Generate ingredients embedding on column {ingredient_columns_map['ingredients']} <<"
        )
        self.__generate_ingredients_embedding(
            column_name=ingredient_columns_map["ingredients"]
        )
        print(">> DONE <<\n")
        print(">> Generate user history ratings <<")
        self.__generate_orders(columns_map=orders_columns_map, rating=rating)
        print(">> DONE <<\n")
        print(">> Generate recipes water footprint dataset <<")
        self.__generate_recipes_wf_category(columns_map=ingredient_columns_map)
        print(">> DONE <<\n")
