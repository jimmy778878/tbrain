import argparse
import ruamel.yaml as yaml
from types import SimpleNamespace


class ArgParser():
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument(
            "--config",
            type=str,
            required=True, 
            help="yaml configuration file path"
        )

    def parse(self):
        args = self.parser.parse_args()
        self.file = args.config
        config = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
        config = self.parse_config(config)
        return config

    def parse_config(self, config: dict):
        parse_result = SimpleNamespace()

        for key, value in config.items():
            if isinstance(value, dict):
                setattr(parse_result, key, self.parse_config(value))
            else:
                setattr(parse_result, key, value)
        
        return parse_result