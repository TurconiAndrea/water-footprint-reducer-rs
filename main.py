"""
Module containing command line utilities to run the recommender from terminal.
"""

import argparse

from cb_recommender import CBRecommender
from cf_recommender import CFRecommender
from configuration import load_configuration
from encoder import Encoder

content_bases_algo = "cb"
collaborative_filtering_algo = "cf"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="- RSWF -",
        usage="%(prog)s [--create-embedding] [--user-id] [--algo] [--filter-wf]",
        description="Recommender System for reducing Water Footprint",
    )
    parser.add_argument(
        "--create-embedding",
        dest="embedding",
        action="store_true",
        help="default False, insert to create the dataset embedding",
    )
    parser.add_argument(
        "--user-id",
        dest="user_id",
        action="store",
        required=True,
        help="the id of the user to recommend recipes",
    )
    parser.add_argument(
        "--algo",
        dest="algorithm",
        action="store",
        required=True,
        help="the algorithm to be used: cb or cf"
    )
    parser.add_argument(
        "--no-filter-wf",
        dest="filter_wf",
        action="store_true",
        help="default False, insert to not recommend recipes without considering water footprint",
    )
    parser.set_defaults(embedding=False)
    parser.set_defaults(filter_wf=False)
    args = parser.parse_args()
    config = load_configuration()

    if args.embedding:
        embedding_encoder = Encoder(config["language"])
        embedding_encoder.generate_data(
            config["orders_map"], config["recipes_map"], bool(config["rating"])
        )

    wf_recommenders = {
        content_bases_algo: CBRecommender(n_recommendations=10, disable_filter_wf=args.filter_wf),
        collaborative_filtering_algo: CFRecommender(n_recommendations=10, disable_filter_wf=args.filter_wf)
    }
    wf_recommender = wf_recommenders[args.algorithm]
    recommendations = wf_recommender.get_user_recommendations(int(args.user_id))

    print("--- RSWF: a Recommender System for reducing Water Footprint ---")
    print(f"--- Recommendation for user with id {args.user_id} ---")
    print(f"--- Total wf: {recommendations['wf'].sum()}")
    print(recommendations)
