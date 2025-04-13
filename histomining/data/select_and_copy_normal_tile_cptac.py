import json
from pathlib import Path
import random
import shutil

import click
import pandas as pd
from tqdm import tqdm


@click.command()
@click.option("--tiles-dir", required=True, help="Directory containing the tiles.")
@click.option(
    "--cptac-luad-csv",
    type=click.Path(exists=True),
    required=True,
    help="Path to the CSV file containing metadata of CPTAC LUAD.",
)
@click.option(
    "--cptac-lusc-csv",
    type=click.Path(exists=True),
    required=True,
    help="Path to the CSV file containing metadata of CPTAC LUSC.",
)
@click.option("--output-dir", required=True, help="Directory to save the selected tiles.")
@click.option(
    "--average-tiles-per-patient",
    default=40,
    help="Average number of tiles per patient.",
)
def main(tiles_dir, cptac_luad_csv, cptac_lusc_csv, output_dir, average_tiles_per_patient):
    """
    Resolution Key for output directory so it should end by one of the following:

    0: 0.5 μm/pixel
    1: 0.6 μm/pixel
    2: 0.7 μm/pixel
    3: 0.8 μm/pixel
    4: 0.9 μm/pixel
    5: 1.0 μm/pixel

    [cancer_type]/[resolution]/[TCGA Barcode]/[region]-[number]-[pixel resolution].jpg
    """

    tiles_dir = Path(tiles_dir)
    output_dir = Path(output_dir)
    if output_dir.name not in ["0", "1", "2", "3", "4", "5"]:
        raise ValueError("Output directory name must be one of the resolution keys.")

    output_dir.mkdir(parents=True, exist_ok=True)

    def filter_df(df):
        return df[(df["Specimen_Type"] == "normal_tissue") & (df["Embedding_Medium"] == "FFPE")]

    luad_df = pd.read_csv(cptac_luad_csv)
    lusc_df = pd.read_csv(cptac_lusc_csv)
    luad_df, lusc_df = filter_df(luad_df), filter_df(lusc_df)
    metadata_df = pd.concat([luad_df, lusc_df], ignore_index=True)
    patient_ids = list(metadata_df["Case_ID"].unique())

    wsi_resolution_mapping = {}
    final_tile_paths = []
    for patient_id in tqdm(patient_ids, desc="Processing patients"):
        wsi_ids = metadata_df.loc[metadata_df["Case_ID"] == patient_id, "Slide_ID"]
        tile_paths = []
        for wsi_id in wsi_ids:
            wsi_dir_matches = list(tiles_dir.glob(f"./*/{wsi_id}/"))

            if len(wsi_dir_matches) != 1:
                print(f"Found {len(wsi_dir_matches)} directories for WSI ID {wsi_id}.")
                continue
            wsi_dir = wsi_dir_matches[0]

            tile_paths.extend(list((wsi_dir / "tiles").glob("*.png")))
            with open(wsi_dir / f"{wsi_id}__metadata.json") as f:
                m = json.load(f)
                resolution = m["x_px_size_tile"]
                if resolution != m["y_px_size_tile"]:
                    print(
                        f"Resolution mismatch for WSI ID "
                        f"{wsi_id}: {resolution} != {m['y_px_size_tile']}"
                    )

            wsi_resolution_mapping[wsi_id] = resolution

        if len(tile_paths) == 0:
            print(f"No tiles found for patient {patient_id}.")
            continue

        if len(tile_paths) < average_tiles_per_patient:
            print(f"Not enough tiles for patient {patient_id}. Found {len(tile_paths)} tiles.")
            continue

        tile_paths = random.sample(tile_paths, k=average_tiles_per_patient)

        final_tile_paths.extend(tile_paths)
    print(f"Total number of tiles selected: {len(final_tile_paths)}")

    tile_count_per_wsi = {}

    for tile_path in tqdm(final_tile_paths, desc="Copying tiles"):
        wsi_id = tile_path.parent.parent.name
        output_wsi_dir = output_dir / wsi_id
        if not output_wsi_dir.exists():
            output_wsi_dir.mkdir()

        tile_count = tile_count_per_wsi.get(wsi_id, 0) + 1
        resolution = wsi_resolution_mapping[wsi_id]
        resolution_str = str(int(resolution * 1000))
        output_path = output_wsi_dir / f"0_{tile_count}_{resolution_str}.jpg"

        shutil.copy(tile_path, output_path)
        tile_count_per_wsi[wsi_id] = tile_count


if __name__ == "__main__":
    main()
