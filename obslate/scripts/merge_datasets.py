import os
import re
import shutil
from pathlib import Path
from typing import List, Union

# from palm.tools.dataset_converter import convert_and_spilt_dataset
from palm.utils.config_utils import get_config_file_path, load_config_to_namespace, load_config
from data_conversion_ufactory_crop import dict_to_namespace, convert_dataset_to_hdf5, get_split_indices, compute_low_dim_mean_and_std


def _numerical_sort(episodes: List[str]) -> List[str]:
    """Sort episode directories numerically based on their extracted episode number."""

    def extract_number(name: str) -> int:
        match = re.search(r"\d+", name)
        return int(match.group()) if match else float("inf")

    return sorted(episodes, key=extract_number)


def merge_datasets(
    d_path1: str,
    d_path2: str,
    output_dir: str,
    num_episodes: List[Union[str, int]],
    config_path: str,
):
    """Merge datasets by copying a specified number of episodes from each dataset."""

    path1, path2, out_path = (
        Path(d_path1),
        Path(d_path2),
        Path(output_dir),
    )
    out_path.mkdir(parents=True, exist_ok=True)

    if not path1.exists() or not path2.exists():
        raise FileNotFoundError("One or both dataset paths do not exist.")

    # List and filter episodes
    episodes_dset1 = [ep for ep in os.listdir(path1) if ep.startswith("episode")]
    episodes_dset2 = [ep for ep in os.listdir(path2) if ep.startswith("episode")]

    sorted_episodes_dset1 = _numerical_sort(episodes_dset1)
    sorted_episodes_dset2 = _numerical_sort(episodes_dset2)

    # Convert 'all' to actual length
    num_episodes[0] = (
        len(sorted_episodes_dset1)
        if num_episodes[0] == "all"
        else min(int(num_episodes[0]), len(sorted_episodes_dset1))
    )
    num_episodes[1] = (
        len(sorted_episodes_dset2)
        if num_episodes[1] == "all"
        else min(int(num_episodes[1]), len(sorted_episodes_dset2))
    )

    selected_episodes = (
        sorted_episodes_dset1[: num_episodes[0]] + sorted_episodes_dset2[: num_episodes[1]]
    )

    # Copy episodes to output directory with sequential naming
    for i, ep in enumerate(selected_episodes):
        source_path = path1 / ep if i < num_episodes[0] else path2 / ep
        target_path = out_path / f"episode{i}"
        shutil.copytree(source_path, target_path)

    # # Convert and split dataset using config
    conversion_config = dict_to_namespace(load_config(config_path))
    f_name = convert_dataset_to_hdf5(output_dir, conversion_config, debug=False)
    get_split_indices(f_name, conversion_config.conversion.train_split_ratio)
    compute_low_dim_mean_and_std(f_name)
    full_config_path = get_config_file_path(config_path)
    shutil.copy(full_config_path, output_dir)

    # Cleanup copied episodes, remove
    for i in range(len(selected_episodes)):
        shutil.rmtree(out_path / f"episode{i}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Merge two datasets by selecting episodes.")
    parser.add_argument("--d1", type=str, required=True, help="Path to the first dataset")
    parser.add_argument("--d2", type=str, required=True, help="Path to the second dataset")
    parser.add_argument("--output", type=str, required=True, help="Output path for merged dataset")
    parser.add_argument(
        "-n",
        "--num_episodes",
        nargs=2,
        required=True,
        help="Number of episodes to merge, e.g., 'all' or an integer",
    )
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")

    args = parser.parse_args()

    # Resolve absolute paths
    script_dir = Path(__file__).resolve().parent
    dataset_dir = script_dir / "../../data"

    args.d1 = str(dataset_dir / args.d1)
    args.d2 = str(dataset_dir / args.d2)
    args.output = str(dataset_dir / args.output)

    merge_datasets(args.d1, args.d2, args.output, args.num_episodes, args.config)
