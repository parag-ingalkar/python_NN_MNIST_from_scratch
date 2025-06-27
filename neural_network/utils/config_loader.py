def load_config(config_path):
    """
    Loads a simple key=value config file into a Python dictionary.

    Args:
        config_path (str): Path to the config file.

    Returns:
        dict: Parsed config values as key-value pairs.
    """
    config = {}
    with open(config_path, "r") as file:
        for line in file:
            line = line.strip()
            if line and "=" in line:
                key, value = line.split("=", 1)
                config[key.strip()] = value.strip()
    return config
