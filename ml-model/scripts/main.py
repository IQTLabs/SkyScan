"""Control logic for running skyscan data and model functionality."""

import configparser


def read_config(config_file="config.ini"):
    """Read in config file values.

    Use python standard library configparser module
    to read in user-defined values. This enables configurability.

    For more info on configparser, see here:
    https://docs.python.org/3/library/configparser.html#module-configparser

    Args:
        config file (str) - name of config file

    Returns:
        config - a config object similar to a dict
    """
    config = configparser.ConfigParser()
    config.read(config_file)

    return config


if __name__ == "__main__":
    pass
