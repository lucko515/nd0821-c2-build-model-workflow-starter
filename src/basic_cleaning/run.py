#!/usr/bin/env python
"""
This step downloads from W&B raw dataset and applies set of data cleaning steps (detecting duplicate rows, converting data types) and exporting the cleaned dataset as a new artifact which is saved to W&B
"""
import argparse
import logging
import wandb

import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Downloading raw dataset from WandB       
    file_path = run.use_artifact(args.input_artifact).file()
    df = pd.read_csv(file_path)

    # Filter out all outlier
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()

    # Convert last_review to datetime
    df['last_review'] = pd.to_datetime(df['last_review'])

    # Handling long-lat boundries
    # NOTE: This can be improved by adding these bountries to config.yaml
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()

    # Save temp file
    df.to_csv(args.output_artifact, index=False)

    # Creating an artifact for the clean dataset
    artifact = wandb.Artifact(
     args.output_artifact,
     type=args.output_type,
     description=args.output_description,
    )
    artifact.add_file(args.output_artifact)

    run.log_artifact(artifact)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Initial step of dataset preparation")


    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="Name of the W&B artifact serving as an input to this artifact",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="Name of the artifact being exported (e.g., clean_data.csv)",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="Artificat type as a lable for W&B (e.g., clean_data)",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="Description of the output artifact",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help="Minimum threshold price of houses in the dataset",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help="Maximum threshold price of houses in the dataset",
        required=True
    )


    args = parser.parse_args()

    go(args)
