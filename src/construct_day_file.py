"""
construct_day_file.py

Batch-constructs test_flow_5000_{index}.txt files (60s epoch format)
from a raw NYC Yellow Taxi CSV, for a list of (date, index) pairs.

Usage:
    python construct_day_file.py \
        --csv data/yellow_tripdata_2016-01.csv \
        --zone-latlong data/ny/zone_latlong.csv \
        --ignorezones data/ny/ignorezonelist.txt \
        --outdir data/ny/files_60sec \
        --map 2016-01-11:21 2016-01-12:22 2016-01-13:23 2016-01-15:24 \
              2016-01-18:25 2016-01-19:26 2016-01-20:27 2016-01-21:28 \
              2016-01-14:29 \
              2016-01-22:30 2016-01-23:31 2016-01-24:32 2016-01-25:33 2016-01-26:34
"""
 
import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

EPOCH_LENGTH_SEC = 60
EPOCHS_PER_DAY = 24 * 3600 // EPOCH_LENGTH_SEC  # 1440

PICKUP_DT_COL = "tpep_pickup_datetime"
PICKUP_LON_COL = "pickup_longitude"
PICKUP_LAT_COL = "pickup_latitude"
DROPOFF_LON_COL = "dropoff_longitude"
DROPOFF_LAT_COL = "dropoff_latitude"

CHUNK_SIZE = 500_000


def parse_map_arg(pairs):
    """--map 2016-01-11:21 2016-01-12:22 ... -> {date_str: index}"""
    date_to_index = {}
    for pair in pairs:
        date_str, idx_str = pair.split(":")
        date_to_index[date_str] = int(idx_str)
    return date_to_index


def load_zone_lookup(path):
    """zone_latlong.csv -> BallTree over (lat, lon) in radians, + zone_id array."""
    df = pd.read_csv(path, header=None, names=["zone_id", "lon", "lat"])
    coords_rad = np.radians(df[["lat", "lon"]].values)
    tree = BallTree(coords_rad, metric="haversine")
    zone_ids = df["zone_id"].values
    return tree, zone_ids


def load_ignore_zones(path):
    with open(path) as f:
        return set(int(line.strip()) for line in f if line.strip())


def match_zones(tree, zone_ids, lat_arr, lon_arr):
    """Nearest-neighbor haversine match -> zone_id array, same length as input."""
    coords_rad = np.radians(np.column_stack([lat_arr, lon_arr]))
    _, idx = tree.query(coords_rad, k=1)
    return zone_ids[idx.flatten()]


def process_date(csv_path, target_date, tree, zone_ids, ignore_zones):
    """
    Stream the month CSV in chunks, filter to target_date (local, naive timestamps
    assumed already America/New_York per TLC convention), match zones, bucket
    into (source, dest, epoch) -> count.
    """
    flow_counts = defaultdict(int)  # (epoch, source_zone, dest_zone) -> count
    target_date_str = target_date  # 'YYYY-MM-DD'

    reader = pd.read_csv(
        csv_path,
        usecols=[PICKUP_DT_COL, PICKUP_LON_COL, PICKUP_LAT_COL,
                  DROPOFF_LON_COL, DROPOFF_LAT_COL],
        parse_dates=[PICKUP_DT_COL],
        chunksize=CHUNK_SIZE,
    )

    for chunk in reader:
        day_mask = chunk[PICKUP_DT_COL].dt.strftime("%Y-%m-%d") == target_date_str
        day_chunk = chunk[day_mask]
        if day_chunk.empty:
            continue

        # drop rows with missing/zero coords (common in this era of TLC data)
        coord_cols = [PICKUP_LAT_COL, PICKUP_LON_COL, DROPOFF_LAT_COL, DROPOFF_LON_COL]
        day_chunk = day_chunk[(day_chunk[coord_cols] != 0).all(axis=1)]
        day_chunk = day_chunk.dropna(subset=coord_cols)
        if day_chunk.empty:
            continue

        pickup_zones = match_zones(
            tree, zone_ids,
            day_chunk[PICKUP_LAT_COL].values, day_chunk[PICKUP_LON_COL].values,
        )
        dropoff_zones = match_zones(
            tree, zone_ids,
            day_chunk[DROPOFF_LAT_COL].values, day_chunk[DROPOFF_LON_COL].values,
        )

        seconds_since_midnight = (
            day_chunk[PICKUP_DT_COL].dt.hour * 3600
            + day_chunk[PICKUP_DT_COL].dt.minute * 60
            + day_chunk[PICKUP_DT_COL].dt.second
        ).values
        epochs = np.minimum(
            seconds_since_midnight // EPOCH_LENGTH_SEC, EPOCHS_PER_DAY - 1
        ).astype(int)

        for src, dst, ep in zip(pickup_zones, dropoff_zones, epochs):
            if src in ignore_zones or dst in ignore_zones:
                continue
            flow_counts[(ep, src, dst)] += 1

    return flow_counts


def write_test_flow_file(flow_counts, out_path):
    """Write in the confirmed format: header=1440, Flows:{e}-{e} marker per epoch
    (including empty ones), then source,dest,count.0 lines."""
    by_epoch = defaultdict(list)
    for (ep, src, dst), count in flow_counts.items():
        by_epoch[ep].append((src, dst, count))

    with open(out_path, "w") as f:
        f.write(f"{EPOCHS_PER_DAY}\n")
        for ep in range(EPOCHS_PER_DAY):
            f.write(f"Flows:{ep}-{ep}\n")
            for src, dst, count in by_epoch.get(ep, []):
                f.write(f"{src},{dst},{float(count)}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="path to month CSV (e.g. yellow_tripdata_2016-01.csv)")
    parser.add_argument("--zone-latlong", required=True)
    parser.add_argument("--ignorezones", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--map", nargs="+", required=True,
                         help="list of DATE:INDEX pairs, e.g. 2016-01-22:30")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    date_to_index = parse_map_arg(args.map)
    tree, zone_ids = load_zone_lookup(args.zone_latlong)
    ignore_zones = load_ignore_zones(args.ignorezones)

    for date_str, index in date_to_index.items():
        print(f"Processing {date_str} -> index {index}")
        flow_counts = process_date(args.csv, date_str, tree, zone_ids, ignore_zones)
        out_path = os.path.join(args.outdir, f"test_flow_5000_{index}.txt")
        write_test_flow_file(flow_counts, out_path)
        print(f"  wrote {out_path} ({len(flow_counts)} non-empty (epoch,src,dst) buckets)")


if __name__ == "__main__":
    main()

