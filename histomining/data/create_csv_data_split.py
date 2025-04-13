from pathlib import Path
import random

import click
import pandas as pd


@click.command()
@click.option("--tcga-ut-path", required=True, help="Path to the TCGA UT directory.")
@click.option("--tcga-ut-csv-split", required=True, help="Path to the TCGA UT CSV split file.")
@click.option("--output-csv", required=True, help="Path to the output CSV file.")
def main(tcga_ut_path, tcga_ut_csv_split, output_csv):
    tcga_ut_path = Path(tcga_ut_path)
    tcga_ut_split = pd.read_csv(tcga_ut_csv_split)

    tile_paths = list((tcga_ut_path / "Lung_normal/0").rglob("*.jpg"))
    patient_ids = list(set([str("-".join(p.parent.name.split("-")[:2])) for p in tile_paths]))
    print(f"Found {len(tile_paths)} tiles from {len(patient_ids)} patients.")

    # Set random seed for reproducibility
    random.seed(42)
    # Shuffle the patient IDs
    random.shuffle(patient_ids)

    # Calculate split sizes
    train_size = int(len(patient_ids) * 0.7)
    val_size = int(len(patient_ids) * 0.15)

    # Split patient IDs into train, val, test sets
    train_patients = patient_ids[:train_size]
    val_patients = patient_ids[train_size : train_size + val_size]
    test_patients = patient_ids[train_size + val_size :]

    print(
        f"Split patients: Train: {len(train_patients)}, Val: {len(val_patients)}, Test: {len(test_patients)}"
    )

    # Create a dictionary to map each patient to its split
    split_dict = {}
    for patient in train_patients:
        split_dict[patient] = "train"
    for patient in val_patients:
        split_dict[patient] = "val"
    for patient in test_patients:
        split_dict[patient] = "test"

    # Create output dataframe
    results = []
    for tile_path in tile_paths:
        patient_id = str("-".join(tile_path.parent.name.split("-")[:2]))
        split = split_dict[patient_id]
        results.append(
            {
                "path": str(tile_path.relative_to(tcga_ut_path)),
                "case": str(tile_path.parents[2].name),
                "patient": patient_id,
                "split_internal": split,
            }
        )

    # Save to CSV
    output_df = pd.DataFrame(results)

    train_tiles = len(output_df[output_df['split_internal'] == 'train'])
    val_tiles = len(output_df[output_df['split_internal'] == 'val'])
    test_tiles = len(output_df[output_df['split_internal'] == 'test'])
    print(f"Split tiles: Train: {train_tiles} ({train_tiles/len(output_df):.1%}), " 
          f"Val: {val_tiles} ({val_tiles/len(output_df):.1%}), "
          f"Test: {test_tiles} ({test_tiles/len(output_df):.1%})")

    output_df["facility"] = None
    output_df["split_external"] = None
    output_df = pd.concat([output_df, tcga_ut_split], axis=0, ignore_index=True)
    output_df.to_csv(output_csv, index=False)
    print(f"Saved split data to {output_csv}")


if __name__ == "__main__":
    main()
