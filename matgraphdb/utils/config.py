import os
import yaml
import re
import logging

import logging.config

from pathlib import Path
import numpy as np
from copy import deepcopy
logger = logging.getLogger(__name__)




class ConfigDict(dict):
    def __init__(self, dictionary: dict):
        dictionary=resolve_templates(dictionary)
        for key, value in dictionary.items():
            if isinstance(value, dict):
                self.__dict__.update({key: ConfigDict(value)})
            else:
                self.__dict__.update({key: value})



    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        if isinstance(value, dict) or isinstance(value, ConfigDict):
            value = getattr(self, key)
            value.update(value)
        setattr(self, key, value)

    def __repr__(self):
        return __class__.__name__ + '(' + repr(self.__dict__) + ')'

    def update(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                self.__dict__.update({key: ConfigDict(value)})
            else:
                self.__dict__.update({key: value})

    def to_dict(self):
        dictionary={}
        for key, value in self.__dict__.items():
            if isinstance(value, ConfigDict):
                dictionary.update({key: value.to_dict()})
            else:
                dictionary.update({key: value})
        return dictionary

    @classmethod
    def from_yaml(cls, yaml_file):
        with open(yaml_file, 'r') as f:
            config = yaml.safe_load(f)
        return cls(config)

    @classmethod
    def from_json(cls, json_file):
        with open(json_file, 'r') as f:
            config = json.load(f)
        return cls(config)

class LoggingConfig(ConfigDict):
    def __init__(self, dictionary: dict):
        super().__init__(dictionary)
        self._update_logger()

    def __setattr__(self, key, value):
        object.__setattr__(key, value)
        self._update_logger()

    def _update_logger(self):
        try:
            logging_config=self.logging_config.to_dict()
            logging_config
            logging.config.dictConfig(self.logging_config.to_dict())
            logging.info(f"Logger configuration updated: {self.logging_config.to_dict()}")
        except Exception as e:
            logging.exception(f"Failed to update logger configuration: {e}")

    def _update_logger(self):
        try:
            logging.config.dictConfig(self.to_dict())
            logging.info(f"Logger configuration updated: {self.to_dict()}")
        except Exception as e:
            logging.error(f"Failed to update logger configuration: {e}")

    def update(self, dictionary):
        super().update(dictionary)
        self._update_logger()


def resolve_templates(d):
    # A function that replaces '{{ var }}' with the actual value in a given string
    def substitute_value(value, mapping):
        # Perform substitution if the value is a string and contains templates
        if isinstance(value, str):
            while True:
                # Find all template variables {{ var }} in the value string
                matches = re.findall(r'\{\{\s*(\w+)\s*\}\}', value)
                if not matches:
                    break  # Exit if no templates found

                # Replace each template with its value from the mapping
                for match in matches:
                    if match in mapping:
                        value = value.replace(f'{{{{ {match} }}}}', mapping[match])
            return value
        return value

    # Recursive function to resolve all template variables in the dictionary
    def resolve_dict(d, mapping):
        resolved_dict = {}
        for key, value in d.items():
            if isinstance(value, dict):
                # Recursively resolve for nested dictionaries
                resolved_dict[key] = resolve_dict(value, mapping)
            else:
                # Substitute the value using the current mapping
                resolved_dict[key] = substitute_value(value, mapping)
            
            # Update the mapping with the resolved value (ensure it resolves for other keys)
            mapping[key] = resolved_dict[key]

        return resolved_dict

    # Initial resolution of templates
    return resolve_dict(d, d.copy())  # Pass a copy of the original dictionary for initial mapping


FILE = Path(__file__).resolve()
PKG_DIR = str(FILE.parents[1])
UTILS_DIR = str(FILE.parents[0])
config = ConfigDict.from_yaml(os.path.join(UTILS_DIR, 'config.yml'))

# config.pkg_dir = PKG_DIR


# print(config.logging_config.loggers)
# config.logging_config.loggers.matgraphdb.level='DEBUG'
# print(config.logging_config.loggers)

logging_config = LoggingConfig(config.logging_config.to_dict())


# print(logging_config.loggers)
# logging_config.loggers.matgraphdb.level='DEBUG'
# print(logging_config.loggers)


if config.log_dir:
    os.makedirs(config.log_dir, exist_ok=True)
if config.data_dir:
    os.makedirs(config.data_dir, exist_ok=True)

# np.set_printoptions(**config.numpy_config.np_printoptions.to_dict())
