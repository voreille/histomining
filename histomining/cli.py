import click


@click.command()
@click.option("--tiles-dir",required=True,type=click.Path(exists=True), help="Root folder of the tiles")
@click.option("--output-json",required=True,type=click.Path(), help="output json file")
def main(tiles_dir):
    """Simple CLI program to greet someone"""
    pass

if __name__ == "__main__":
    main()