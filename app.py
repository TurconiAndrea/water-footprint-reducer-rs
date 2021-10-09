import pandas as pd
import streamlit as st

from configuration import load_configuration
from cb_recommender import CBRecommender


class App:
    def __init__(self):
        config = load_configuration()
        self.path_orders = config["path_orders"]
        self.path_recipes = config["path_recipes"]

    @st.cache
    def load_recipes(self):
        return pd.read_pickle(self.path_recipes)

    @st.cache
    def load_orders(self):
        return pd.read_pickle(self.path_orders)

    def generate_user_orders(self, user_id, orders, recipes):
        user_orders = orders.query(f"user_id == {user_id}")
        return pd.merge(user_orders, recipes, on=["id"])[
            ["name", "rating", "wf", "category"]
        ]

    def recommend_recipes(self, user_id, n_recommendations, filter_wf):
        wf_recommender = CBRecommender(
            n_recommendations=n_recommendations, filter_wf=not filter_wf
        )
        return wf_recommender.get_user_recommendations(user_id)

    def build_app(self):
        recipes = self.load_recipes()
        orders = self.load_orders()

        st.markdown("# Recommender System for reducing Water footprint ")

        st.sidebar.write("### Configure the system")
        user_id = st.sidebar.selectbox(
            "Select a user", orders["user_id"].unique(), index=0
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
            with st.spinner(text="Generating recoomendations"):
                st.markdown("### Recommendations for user to lower the Water Footprint")
                recommendations = self.recommend_recipes(
                    user_id, n_recommendations, filter_wf
                )[["name", "wf", "category"]].reset_index(drop=True)
                st.write(
                    "Total Water footprint of recommendations:",
                    round(recommendations["wf"].sum(), 2),
                )
                st.write(recommendations)


if __name__ == "__main__":
    app = App()
    app.build_app()
