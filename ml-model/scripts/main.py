"""Control logic for running skyscan data and model functionality."""

import argparse
import configparser


def read_config(config_file="config.ini"):
    """Read in config file values.

    Use python standard library configparser module
    to read in user-defined values. This enables configurability.

    For more info on configparser, see here:
    https://docs.python.org/3/library/configparser.html#module-configparser

    Args:
        config file (str) - name of config file. Default is config.ini

    Returns:
        config - a config object similar to a dict
    """
    config = configparser.ConfigParser()
    config.read(config_file)

    return config


def parse_command_line_arguments():
    """Parse command line arguments with argparse."""
    # TODO: incomplete.
    parser = argparse.ArgumentParser(
        description="Run skyscan data and model scripts.",
        epilog="For help with this program, contact John Speed at jmeyers@iqt.org.",
    )
    argparse.ArgumentParser("--prepare_dataset", description="Prepare voxel51 dataset.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_command_line_arguments()
