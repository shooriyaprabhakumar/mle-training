import os
import tarfile
import urllib.request
import pandas as pd

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    """
    Main function to fetch and load housing data, then save it to a CSV file.

    Parameters
    ----------
    output_folder : str
        The folder to save the ingested data.

    Returns
    -------
    None
    """

    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def main(output_folder):
    fetch_housing_data()
    housing = load_housing_data()
    housing.to_csv(os.path.join(output_folder, "housing_data.csv"), index=False)
    print(f"Data ingested and saved to {output_folder}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-folder",
        type=str,
        required=True,
        help="Folder to save the ingested data",
    )
    args = parser.parse_args()
    main(args.output_folder)
