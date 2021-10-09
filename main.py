import argparse

from configuration import load_configuration
from cb_recommender import CB_Recommender
from encoder import Encoder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="- RSWF -",
        usage="%(prog)s [--create-embedding] [--user-id] [--filter-wf]",
        description="Recommender System for reducing Water Footprint",
    )
    parser.add_argument(
        "--create-embedding",
        dest="embedding",
        action="store_true",
        help="defualt False, insert to create the dataset embedding",
    )
    parser.add_argument(
        "--user-id",
        dest="user_id",
        action="store",
        required=True,
        help="the id of the user to recommend recipes",
    )
    parser.add_argument(
        "--no-filter-wf",
        dest="filter_wf",
        action="store_true",
        help="default False, insert to not recommemd recipes without considering water footprint",
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

    wf_recommender = CB_Recommender(filter_wf=args.filter_wf)
    recommendations = wf_recommender.get_user_recommendations(args.user_id)
    print("--- RSWF: a Recommender System for reducing Water Footprint ---")
    print(f"--- Recommendation for user with id {args.user_id} ---")
    # print(f"--- Total wf: {recommendations['wf'].sum()}")
    print(recommendations)
