"""
Module containing the Streamlit app to explore the project.
"""

import pandas as pd
import streamlit as st

from cb_recommender import CBRecommender
from cf_recommender import CFRecommender
from configuration import load_configuration

content_bases_algo = "Content Based"
collaborative_filtering_algo = "Collaborative filtering"


class App:
    """
    Class that represents the app of the recommender system.
    It provides a Streamlit app to run on local browser
    and the possibility to configure the system with the
    two different algorithm and the water footprint filter.
    """

    def __init__(self):
        """
        Constructor method for the class.
        It loads the paths for the orders and the recipes datasets
        """
        config = load_configuration()
        self.path_orders = config["path_orders"]
        self.path_recipes = config["path_recipes"]

    @st.cache
    def load_recipes(self):
        """
        Load the recipes dataset from the default folder.

        :return: a dataframe containing the recipes.
        """
        return pd.read_pickle(self.path_recipes)

    @st.cache
    def load_orders(self):
        """
        Load the orders from the default folder.

        :return: a dataframe containing the orders and ratings.
        """
        return pd.read_pickle(self.path_orders)

    def generate_user_orders(self, user_id, orders, recipes):
        """
        Return the user orders ratings merged with the recipes
        water footprint information.

        :param user_id: the id of the user.
        :param orders: the dataframe containing orders.
        :param recipes: the dataframe containing recipes.
        :return: a dataframe with user orders ratings and
            recipe water footprint information.
        """
        user_orders = orders.query(f"user_id == {user_id}")
        return pd.merge(user_orders, recipes, on=["id"])[
            ["name", "rating", "wf", "category"]
        ]

    def get_recipe_recommendations(
        self, user_id, n_recommendations, filter_wf, algo_type
    ):
        """
        Get recipe recommendations for provided user using
        one of the two algorithm and the water footprint filter.

        :param user_id: the id of the user to recommend recipes.
        :param n_recommendations: the number of recommendations.
        :param filter_wf: the information about the activation of
            water footprint filter or not.
        :param algo_type: the type of the recommendation algorithm.
        :return: a dataframe containing user recommendations.
        """
        wf_recommenders = {
            content_bases_algo: CBRecommender(
                n_recommendations=n_recommendations, disable_filter_wf=not filter_wf
            ),
            collaborative_filtering_algo: CFRecommender(
                n_recommendations=n_recommendations, disable_filter_wf=not filter_wf
            ),
        }
        wf_recommender = wf_recommenders[algo_type]
        return wf_recommender.get_user_recommendations(user_id)

    def build_app(self):
        """
        Build the app with Streamlit package.
        App provides some configuration like the id of the user
        to recommend recipes, the possibility to choose between
        a content based or a collaborative filtering algorithm
        and the possibility to activate or deactivate the
        water footprint filter.

        :return: None.
        """
        recipes = self.load_recipes()
        orders = self.load_orders()

        st.markdown("# Recommender System for reducing Water footprint ")

        st.sidebar.write("### Configure the system")
        user_id = st.sidebar.selectbox(
            "Select a user", orders["user_id"].unique(), index=0
        )
        algo_type = st.sidebar.selectbox(
            "Select the algorithm for recommendation",
            [content_bases_algo, collaborative_filtering_algo],
            index=0,
        )
        n_recommendations = st.sidebar.slider(
            "Select number of recommendations",
            min_value=1,
            max_value=20,
            value=10,
            step=1,
        )
        st.sidebar.markdown("***")
        st.sidebar.write(
            "Water Footprint filter is activate by default, check the option to change"
        )
        filter_wf = st.sidebar.checkbox(
            "Deactivate the Water Footprint filter", value=True
        )

        st.markdown("### User rating for recipes")
        st.dataframe(self.generate_user_orders(user_id, orders, recipes))

        if st.button("Recommend recipes!"):
            with st.spinner(text="Generating recommendations"):
                st.markdown("### Recommendations for user to lower the Water Footprint")
                recommendations = self.get_recipe_recommendations(
                    user_id, n_recommendations, filter_wf, algo_type
                )[["name", "wf", "category"]].reset_index(drop=True)
                st.write(
                    "Total Water footprint of recommendations:",
                    round(recommendations["wf"].sum(), 2),
                )
                st.dataframe(recommendations)


if __name__ == "__main__":
    app = App()
    app.build_app()
