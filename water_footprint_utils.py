import pandas as pd

from configuration import load_configuration


class WaterFootprintUtils:
    def __init__(self):
        config = load_configuration()
        self.orders = pd.read_pickle(config["path_orders"])
        self.recipes = pd.read_pickle(config["path_recipes"])
        self.classes = ["A", "B", "C", "D", "E"]

    def __get_recipe_class_to_recommend(self, user_score):
        usr_cls = self.classes.index(user_score)
        usr_cls_1 = usr_cls - 1 if usr_cls - 1 >= 0 else 0
        usr_cls_2 = usr_cls - 2 if usr_cls - 1 >= 1 else usr_cls
        return sorted({"A", self.classes[usr_cls_1], self.classes[usr_cls_2]})

    def __get_recipe_class(self, id):
        category = self.recipes.query(f"id == {id}")["category"].tolist()
        return category[0] if category else None

    def __get_user_score(self, user_id, weight=1.3):
        user_df = self.orders.query(f"user_id == {user_id}").copy()
        user_df["category"] = user_df["id"].apply(lambda x: self.__get_recipe_class(x))
        orders = user_df.groupby(by="category").count()["id"].to_dict()
        total = sum(orders[k] for k in orders)
        orders = {k: round((orders[k] / total) * 100, 2) for k in orders}
        e_percentage = orders["E"] if "E" in orders else 0
        d_percentage = orders["D"] if "D" in orders else 0
        a_percentage = orders["A"] if "A" in orders else 0
        b_percentage = orders["B"] if "B" in orders else 0
        diff = (a_percentage + b_percentage) - (e_percentage + d_percentage) * weight
        user_cls = "D"
        if -5 <= diff <= 5:
            user_cls = "C"
        elif 5 < diff <= 25:
            user_cls = "B"
        elif diff > 25:
            user_cls = "A"
        elif -25 <= diff < -5:
            user_cls = "D"
        elif diff < -25:
            user_cls = "E"
        return user_cls

    def __get_recipe_from_id(self, recipe_id):
        recipe = self.recipes.query(f"id == {recipe_id}")["category"].tolist()
        return recipe[0] if recipe else "F"

    def get_recommendations_correct(self, recommendations, user_id, algo_type):
        user_score = self.__get_user_score(user_id)
        class_to_rec = self.__get_recipe_class_to_recommend(user_score)
        return [
            rec
            for rec in recommendations
            if self.recipes["category"][rec] in class_to_rec
        ] if algo_type == "cb" else [
            recipe_id
            for recipe_id in recommendations
            #if self.recipes.query(f"id == {recipe_id}")["category"].tolist()[0] in class_to_rec
            if self.__get_recipe_from_id(recipe_id) in class_to_rec
        ]


if __name__ == "__main__":
    wf = WaterFootprintUtils()
