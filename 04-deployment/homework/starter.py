#!/usr/bin/env python
# coding: utf-8
import sys
import pickle
import pandas as pd


with open("model.bin", "rb") as f_in:
    dv, lr = pickle.load(f_in)


def read_data(filename, categorical):
    df = pd.read_parquet(filename)

    df["duration"] = df.dropOff_datetime - df.pickup_datetime
    df["duration"] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype("int").astype("str")

    return df


def main():
    year = int(sys.argv[1])
    month = int(sys.argv[2])

    categorical = ["PUlocationID", "DOlocationID"]
    df = read_data(
        f"https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{year:04d}-{month:02d}.parquet",
        categorical,
    )

    dicts = df[categorical].to_dict(orient="records")
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    print(f"mean prediction is = {y_pred.mean()}")
    df["ride_id"] = f"{year:04d}/{month:02d}_" + df.index.astype("str")

    df_results = pd.DataFrame()
    df_results["ride_id"] = df["ride_id"]
    df_results["duration_prediction"] = y_pred

    output_file = f"tmp/fhv_tripdata_{year:04d}_{month:02d}.parquet"

    df_results.to_parquet(output_file, engine="pyarrow", compression=None, index=False)


if __name__ == "__main__":
    main()
