from pathlib import Path

import click
import h5py
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from histomining.data.torch_datasets import EmbeddingDataset
from histomining.models.linear_probing import LinearProbingFromEmbeddings


def split_data(embeddings, paths, labels, split_df):
    """Split data into train, val, and test sets."""

    # Create a DataFrame from the embeddings
    split_df = split_df.set_index("path")
    train_paths = split_df[split_df["split_internal"] == "train"].index
    val_paths = split_df[split_df["split_internal"] == "val"].index
    test_paths = split_df[split_df["split_internal"] == "test"].index

    # Find indices for each split
    train_indices = [i for i, path in enumerate(paths) if path in train_paths]
    val_indices = [i for i, path in enumerate(paths) if path in val_paths]
    test_indices = [i for i, path in enumerate(paths) if path in test_paths]

    # Extract embeddings and labels for each split
    train_embeddings = embeddings[train_indices]
    val_embeddings = embeddings[val_indices]
    test_embeddings = embeddings[test_indices]

    train_labels = labels[train_indices]
    val_labels = labels[val_indices]
    test_labels = labels[test_indices]

    print(f"Train: {len(train_indices)} samples")
    print(f"Val: {len(val_indices)} samples")
    print(f"Test: {len(test_indices)} samples")

    return (
        train_embeddings,
        train_labels,
        val_embeddings,
        val_labels,
        test_embeddings,
        test_labels,
    )


def get_dataloader(
    embeddings,
    tile_paths,
    labels,
    split_df,
    batch_size=32,
    num_workers=0,
):
    """Create a DataLoader from embeddings and labels."""

    train_embeddings, train_labels, val_embeddings, val_labels, test_embeddings, test_labels = (
        split_data(embeddings, tile_paths, labels, split_df)
    )
    train_dataloader = DataLoader(
        EmbeddingDataset(train_embeddings, train_labels),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    val_dataloader = DataLoader(
        EmbeddingDataset(val_embeddings, val_labels),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    test_dataloader = DataLoader(
        EmbeddingDataset(test_embeddings, test_labels),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_dataloader, val_dataloader, test_dataloader


def load_embeddings(path, label_map=None):
    if label_map is None:
        label_map = {"NORMAL": 0, "LUAD": 1}

    with h5py.File(path, "r") as f:
        # Load embeddings as numpy array
        embeddings = f["embeddings"][:]  # Shape: (num_samples, embedding_dim)

        # Load metadata
        tile_paths = f["paths"][:]  # List of strings
        labels = f["labels"][:]  # List of strings

        # Load attributes
        embedding_dim = f.attrs["embedding_dim"]
        model_name = f.attrs["model_name"]
    tile_paths = [str(p, encoding="utf-8") for p in tile_paths]
    labels = np.array([label_map[str(label, encoding="utf-8")] for label in labels])
    return embeddings, tile_paths, labels, embedding_dim, model_name


@click.command()
@click.option("--output-path", default=None, help="Path to store the weight of the linear layer.")
@click.option("--embeddings-h5-path", required=True, help="Path to the H5 file with embeddings.")
@click.option(
    "--split-csv-path", required=True, help="Path to the CSV file with split information."
)
@click.option("--num-epochs", default=10, help="Number of epochs to train.")
@click.option("--gpu-id", default=0, help="Name of the model to use.")
@click.option("--batch-size", default=256, help="Batch size for inference.")
@click.option("--lr", default=0.001, show_default=True, help="LR for Adam")
@click.option(
    "--weight-decay", default=0, type=click.FLOAT, show_default=True, help="Weight decay for Adam"
)
@click.option("--num-workers", default=0, help="Number of workers for dataloader.")
def main(
    output_path,
    embeddings_h5_path,
    split_csv_path,
    num_epochs,
    gpu_id,
    batch_size,
    lr,
    weight_decay,
    num_workers,
):
    """Simple CLI program to greet someone"""
    embeddings_h5_path = Path(embeddings_h5_path).resolve()

    embeddings, tile_paths, labels, embedding_dim, model_name = load_embeddings(embeddings_h5_path)
    split_df = pd.read_csv(split_csv_path)

    train_loader, val_loader, test_loader = get_dataloader(
        embeddings,
        tile_paths,
        labels,
        split_df,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    linear_probing = LinearProbingFromEmbeddings(
        embedding_dim,
        num_classes=2,
        lr=lr,
        weight_decay=weight_decay,
    )
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator="gpu",
        devices=[gpu_id],
        # log_every_n_steps=log_every_n,
        precision="16-mixed",
        logger=TensorBoardLogger("tb_logs", name="linear_probing_from_embeddings"),
        # check_val_every_n_epoch=10,
    )
    trainer.logger.log_hyperparams(
        {
            "run/embeddings_h5_path": embeddings_h5_path,
            "run/output_path": output_path,
            "run/gpu_id": gpu_id,
            "run/num_epochs": num_epochs,
            "run/num_workers": num_workers,
            "run/module_class": linear_probing.__class__.__name__,
        }
    )
    trainer.fit(linear_probing, train_loader, val_loader)
    trainer.test(linear_probing, test_loader)
    if output_path:
        trainer.save_checkpoint(output_path)


if __name__ == "__main__":
    main()
