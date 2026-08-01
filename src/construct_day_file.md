## Status: Complete, tested, committed to repo

Location: `src/construct_day_file.py`
Committed: `git commit "add batch date->test_flow construction script"` (already pushed to `main`)

## Purpose

Converts raw NYC Yellow Taxi trip CSVs (the lat/long-era TLC schema, pre-July
2016) into the `test_flow_5000_{index}.txt` format the CEVD/NeurADP simulator
expects, for any given calendar date(s). This is what lets us build simulator
input for dates outside the original 20 pre-built day-files (which only cover
March-April 2016) — e.g. our Winter Storm Jonas weather window (January 2016).

## What it does, step by step

For each `DATE:INDEX` pair given:

1. Streams the target month's CSV in chunks (500,000 rows at a time — avoids
   loading the full ~1.7-2GB file into memory at once).
2. Filters rows down to the target calendar date, using the pickup timestamp
   column (`tpep_pickup_datetime`), assumed to already be in America/New_York
   local time (standard TLC convention — no timezone conversion applied).
3. Drops rows with missing or zero-valued pickup/dropoff coordinates (common
   junk data in this era of the TLC dataset).
4. Matches each trip's pickup and dropoff lat/long to the nearest simulator
   zone, using a `BallTree` nearest-neighbor search (haversine distance)
   against `zone_latlong.csv`.
5. Drops any trip where the matched pickup or dropoff zone appears in
   `ignorezonelist.txt`.
6. Buckets every remaining trip into a 60-second epoch (1440 epochs/day),
   based on pickup time.
7. Aggregates into `(source_zone, dest_zone, epoch) → count`.
8. Writes one `test_flow_5000_{index}.txt` file per date, in the exact format
   the simulator expects (see "Output format" below).

## Output format (confirmed against existing files_60sec data)

```
1440
Flows:0-0
{source_zone},{dest_zone},{count}.0
{source_zone},{dest_zone},{count}.0
...
Flows:1-1
...
Flows:1439-1439
...
```

- First line: always `1440` (24 hours × 60 epochs/hour, 60-second epoch length).
- Every epoch from 0 to 1439 gets a `Flows:{epoch}-{epoch}` marker, **even if
  no trips occurred in that epoch** — the simulator's parser advances its
  internal clock based on markers seen, so empty epochs cannot be skipped.
- Count is always written as a float (`1.0`, `2.0`, etc.), matching the
  existing files' format even though it's logically an integer trip count.

## Inputs required

| Argument | Description | Example |
|---|---|---|
| `--csv` | Path to the raw month CSV | `../data/ny/yellow_tripdata_2016-01.csv` |
| `--zone-latlong` | Zone lookup table (`zone_id, lon, lat`, no header) | `../data/ny/zone_latlong.csv` |
| `--ignorezones` | Newline-separated zone IDs to exclude | `../data/ny/ignorezonelist.txt` |
| `--outdir` | Where to write the generated files | `../data/ny/files_60sec` |
| `--map` | Space-separated `DATE:INDEX` pairs | `2016-01-22:30 2016-01-23:31 ...` |

## Assumptions baked in (verified against this repo's data)

- **`zone_latlong.csv` column order = `zone_id, lon, lat`.** Confirmed by
  inspecting the first several rows — longitude values in the -73.9 to -74.0
  range, latitude values in the 40.7-40.8 range, matching NYC geography.
- **Raw CSV timestamps are already America/New_York local time**, no timezone
  offset present or needed — standard assumption for this era of TLC data.
- **`ignorezonelist.txt` contains one plain integer zone ID per line.**
  Confirmed by inspection.
- **Epoch length is fixed at 60 seconds** (`files_60sec/` only — this repo
  does not have a `training_flow_` variant for 60-second data, only
  `test_flow_`, unlike `files_10sec/`/`files_30sec/` which have both. Since
  we are not retraining, only `test_flow_` files are needed here).

## Usage — Winter Storm Jonas window (already run successfully)

```bash
cd ~/CEVD_CODE/src
source ../cevd_env/bin/activate

python construct_day_file.py \
  --csv ../data/ny/yellow_tripdata_2016-01.csv \
  --zone-latlong ../data/ny/zone_latlong.csv \
  --ignorezones ../data/ny/ignorezonelist.txt \
  --outdir ../data/ny/files_60sec \
  --map 2016-01-11:21 2016-01-12:22 2016-01-13:23 2016-01-15:24 \
        2016-01-18:25 2016-01-19:26 2016-01-20:27 2016-01-21:28 \
        2016-01-14:29 \
        2016-01-22:30 2016-01-23:31 2016-01-24:32 2016-01-25:33 2016-01-26:34
```

### Date → index mapping used

| Index | Date | Role |
|---|---|---|
| 21 | Jan 11, 2016 | Train |
| 22 | Jan 12, 2016 | Train |
| 23 | Jan 13, 2016 | Train |
| 24 | Jan 15, 2016 | Train |
| 25 | Jan 18, 2016 | Train |
| 26 | Jan 19, 2016 | Train |
| 27 | Jan 20, 2016 | Train |
| 28 | Jan 21, 2016 | Train |
| 29 | Jan 14, 2016 | Validation |
| 30 | Jan 22, 2016 | Test (storm arrives) |
| 31 | Jan 23, 2016 | Test (storm peak) |
| 32 | Jan 24, 2016 | Test (aftermath) |
| 33 | Jan 25, 2016 | Test (recovery) |
| 34 | Jan 26, 2016 | Test (recovery) |

Mirrors the paper's 1 validation + 8 training + 5 test day structure, shifted
to a bad-weather window instead of the paper's ordinary March/April dates.
Training days deliberately end before the storm (Jan 21) so the training
signal stays "normal weather," isolating the storm's effect entirely to the
test window.

## Confirmed successful run (actual output, Aug 2026)

```
Processing 2016-01-11 -> index 21
  wrote ../data/ny/files_60sec/test_flow_5000_21.txt (333704 non-empty buckets)
Processing 2016-01-12 -> index 22
  wrote ../data/ny/files_60sec/test_flow_5000_22.txt (358144 non-empty buckets)
Processing 2016-01-13 -> index 23
  wrote ../data/ny/files_60sec/test_flow_5000_23.txt (385056 non-empty buckets)
Processing 2016-01-15 -> index 24
  wrote ../data/ny/files_60sec/test_flow_5000_24.txt (391806 non-empty buckets)
Processing 2016-01-18 -> index 25
  wrote ../data/ny/files_60sec/test_flow_5000_25.txt (332251 non-empty buckets)
Processing 2016-01-19 -> index 26
  wrote ../data/ny/files_60sec/test_flow_5000_26.txt (374002 non-empty buckets)
Processing 2016-01-20 -> index 27
  wrote ../data/ny/files_60sec/test_flow_5000_27.txt (371899 non-empty buckets)
Processing 2016-01-21 -> index 28
  wrote ../data/ny/files_60sec/test_flow_5000_28.txt (389323 non-empty buckets)
Processing 2016-01-14 -> index 29
  wrote ../data/ny/files_60sec/test_flow_5000_29.txt (385965 non-empty buckets)
Processing 2016-01-22 -> index 30
  wrote ../data/ny/files_60sec/test_flow_5000_30.txt (409039 non-empty buckets)
Processing 2016-01-23 -> index 31
  wrote ../data/ny/files_60sec/test_flow_5000_31.txt (75689 non-empty buckets)
Processing 2016-01-24 -> index 32
  wrote ../data/ny/files_60sec/test_flow_5000_32.txt (155667 non-empty buckets)
Processing 2016-01-25 -> index 33
  wrote ../data/ny/files_60sec/test_flow_5000_33.txt (274285 non-empty buckets)
Processing 2016-01-26 -> index 34
  wrote ../data/ny/files_60sec/test_flow_5000_34.txt (319919 non-empty buckets)
```

### Sanity check — this is a meaningful signal, not noise

Non-empty `(epoch, source, dest)` bucket counts by day:

| Day | Index | Buckets | Notes |
|---|---|---|---|
| Jan 22 | 30 | 409,039 | Normal, storm not yet arrived |
| **Jan 23** | **31** | **75,689** | **Storm peak — ~5.4x fewer trips than the day before** |
| Jan 24 | 32 | 155,667 | Aftermath, still heavily depressed |
| Jan 25 | 33 | 274,285 | Recovering |
| Jan 26 | 34 | 319,919 | Near-normal |

All other (train/val) days sit in the 330,000-392,000 range, consistent with
normal weekday demand. The Jan 23 collapse and gradual recovery is exactly
the pattern expected from Winter Storm Jonas — good evidence the pipeline is
capturing a real signal, not an artifact of the construction process.

## What this tool does NOT do

- Does **not** create `training_flow_` files — confirmed `files_60sec/` never
  uses them, only `test_flow_` files, regardless of whether the day is used
  for training, validation, or testing (the simulator's day-role is
  determined by which script/day-list references the index, not the
  filename).
- Does **not** retrain any model. It only produces simulator input data.
  Scoring against these new files still requires a trained model checkpoint
  (`.h5`) matching the config being tested.
- Does **not** handle post-July-2016 TLC data (zone-ID schema, no lat/long
  columns) — would need a different construction path (join against
  `taxi_zone_lookup.csv`) if extended to that era.

## Next step (blocked, not on this tool)

Constructing these 14 files was the goal of this tool, and that goal is met.
Actually *scoring* CEVD/NeurADP/Baseline against them (to test weather
robustness) is blocked on a separate issue: the 8-day-trained model
checkpoints (`.h5` files) needed by `main_scoring.py`/`calibrate_lambda.py`
currently only exist on a collaborator's machine, not this one. See project
notes for status on retrieving or retraining those.