import pandas as pd

from configuration import load_configuration


class WaterFootprintUtils:
    """
    Class that represent utilities for the water footprint
    reduction. This class provides a method for computing
    the user score based on his reviews and orders.
    It also provides a method for reducing the given
    recommendations for the user.
    """
    def __init__(self):
        config = load_configuration()
        self.orders = pd.read_pickle(config["path_orders"])
        self.recipes = pd.read_pickle(config["path_recipes"])
        self.classes = ["A", "B", "C", "D", "E"]

    def __get_recipe_class_to_recommend(self, user_score):
        """
        Get the recipe categories to recommend based on the user score.
        The category are always the two before the user score and
        recipes with category equals to A.

        :param user_score: the score of the user.
        :return: a list containing the categories of the recipe to
            recommend.
        """
        usr_cls = self.classes.index(user_score)
        usr_cls_1 = usr_cls - 1 if usr_cls - 1 >= 0 else 0
        usr_cls_2 = usr_cls - 2 if usr_cls - 1 >= 1 else usr_cls
        return sorted({"A", self.classes[usr_cls_1], self.classes[usr_cls_2]})

    def __get_recipe_class(self, recipe_id):
        """
        Get the category of the recipe from its id.

        :param recipe_id: the id of the recipe.
        :return: the category of the recipe if exists.
        """
        category = self.recipes.query(f"id == {recipe_id}")["category"].tolist()
        return category[0] if category else None

    def __get_user_score(self, user_id, weight=1.3):
        """
        Get the score of the user based on his reviews.
        User orders are summed and weighted based on their
        categories. Then based on the result the user score
        is found.

        :param user_id: the id of the user.
        :param weight: the weight for the recipe categories.
        :return: the user score.
        """
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

    def __get_recipe_category(self, recipe_id):
        """
        Return the category of the recipe row from the
        dataframe based on the recipe id.

        :param recipe_id: the id of the recipe.
        :return: the category of the recipe at the provided id.
        """
        recipe = self.recipes.query(f"id == {recipe_id}")["category"].tolist()
        return recipe[0] if recipe else "F"

    def get_recommendations_correct(self, recommendations, user_id, algo_type):
        """
        Get the correct recipe recommendations from a list of
        recommendations ids based on the user score and the
        type of the algorithm.

        :param recommendations: a list containing all the recommended recipes.
        :param user_id: the id of the user.
        :param algo_type: the type of the algorithm
            (Content Based or Collaborative Filtering)
        :return: a list containing all the recipes filtered by
            water footprint.
        """
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
            if self.__get_recipe_category(recipe_id) in class_to_rec
        ]


if __name__ == "__main__":
    wf = WaterFootprintUtils()
