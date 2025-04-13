from pathlib import Path

import click
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from histomining.data.torch_datasets import TileDataset
from histomining.models.foundation_models import load_model
from histomining.utils import get_device

label_map = {
    "Lung_normal": "NORMAL",
    "Lung_adenocarcinoma": "LUAD",
}


@click.command()
@click.option("--data-dir", required=True, help="Directory containing the data.")
@click.option("--model-name", default="UNI2", help="Name of the model to use.")
@click.option("--output-file", required=True, help="Path to output H5 file.")
@click.option("--batch-size", default=256, help="Batch size for processing.")
@click.option("--num-workers", default=32, help="Number of workers for DataLoader.")
@click.option("--gpu-id", default=0, help="GPU ID to use.")
@click.option("--magnification-key", default=0, help="Magnification key as defined by TCGA-UT.")
def main(data_dir, model_name, output_file, batch_size, num_workers, gpu_id, magnification_key):
    """Simple CLI program to greet someone"""
    data_dir = Path(data_dir)
    tile_paths = list(data_dir.glob(f"./*/{magnification_key}/*/*.jpg"))
    tile_ids = [str(p.parent.name + "/" + p.name) for p in tile_paths]
    relative_tile_paths = [str(p.relative_to(data_dir)) for p in tile_paths]
    labels = [label_map[tile_path.parents[2].name] for tile_path in tile_paths]

    device = get_device(gpu_id)
    model, preprocess, embedding_dim, autocast_dtype = load_model(
        model_name, device, apply_torch_scripting=True
    )

    dataset = TileDataset(
        tile_paths,
        preprocess=transforms.Compose(
            [
                transforms.CenterCrop(224),
                preprocess,
            ]
        ),
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True
    )

    embeddings = []
    with torch.autocast(device_type=device.type, dtype=autocast_dtype):
        with torch.inference_mode():
            for images in tqdm(dataloader, desc="Processing batches"):
                images = images.to(device)
                batch_embeddings = model(images)
                embeddings.append(batch_embeddings.cpu().numpy())
            # Process embeddings as needed
    embeddings = np.concatenate(embeddings, axis=0)

    # Save to H5 file
    with h5py.File(output_file, "w") as f:
        # Create datasets
        f.create_dataset("embeddings", data=embeddings, dtype="float32")
        # Store strings as variable-length ASCII
        dt_str = h5py.special_dtype(vlen=str)
        f.create_dataset("tile_ids", data=tile_ids, dtype=dt_str)
        f.create_dataset("paths", data=relative_tile_paths, dtype=dt_str)
        f.create_dataset("labels", data=labels, dtype=dt_str)

        # Store metadata
        f.attrs["embedding_dim"] = embedding_dim
        f.attrs["model_name"] = model_name
        f.attrs["num_samples"] = len(tile_paths)


if __name__ == "__main__":
    main()
