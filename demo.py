import random

import pandas as pd
import streamlit as st
from configuration import load_configuration


class Demo:
    def __init__(self):
        config = load_configuration()
        self.path_recipes = config["path_recipes"]

    @st.cache
    def load_recipes(self):
        """
        Load the recipes dataset from the default folder.

        :return: a dataframe containing the recipes.
        """
        return pd.read_pickle(self.path_recipes)

    def get_recipe_info(self, name, recipes):
        return recipes.query(f'name == "{name}"')

    def display_recipes_wf(self, recipes_info):
        data = {"id": [], "wf": []}
        for idx, col in enumerate(st.columns(3)):
            recipe = recipes_info[idx].to_dict("records")[0]
            col.metric(recipe["name"].capitalize(), f"{recipe['wf']} l", delta=recipe["category"])
            data["id"].append(idx)
            data["wf"].append(float(recipe["wf"]))

        st.bar_chart(pd.DataFrame(data))

    def build_demo(self):
        recipes = self.load_recipes()

        st.markdown("# 🍕 Recipes Water Footprint Tool")
        st.markdown("This tools will provide information about recipes that you are going to choose.\n"
                    "Also it will suggest other recipes that will help you lower water footprint")
        random.seed(10)
        random_recipes = random.sample(recipes["name"].unique().tolist(), 15)
        choices = st.multiselect("Choose 3 recipes:", random_recipes)

        if len(choices) != 3:
            st.error("Please, select 3 recipes")
        else:
            recipes_info = [self.get_recipe_info(recipe, recipes) for recipe in choices]
            st.markdown("### Your recipes water footprint")
            self.display_recipes_wf(recipes_info)


if __name__ == '__main__':
    demo = Demo()
    demo.build_demo()
