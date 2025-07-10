import yaml 


def load_yaml_file(config_file):
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)
    return config