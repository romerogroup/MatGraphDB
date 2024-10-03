import os
import yaml
import re
import logging

from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


class ConfigDict:
    def __init__(self, dictionary:dict, root_config=None):
        """
        Initialize the ConfigDict with a dictionary.

        Parameters
        ----------
        dictionary : dict
            The dictionary to be wrapped for attribute-style access.
        root_config : Config, optional
            The root configuration object to save changes to, by default None.
        """
        self._dict = dictionary
        self._root_config = root_config or self
        logger.debug(f"ConfigDict initialized with {dictionary}")

    def __getattr__(self, name:str):
        """
        Get a value from the dictionary or raise AttributeError if not found.

        Parameters
        ----------
        name : str
            The key to look for in the dictionary.

        Returns
        -------
        Any
            The value corresponding to the key in the dictionary.

        Examples
        --------
        >>> config_dict = ConfigDict({'key': 'value'})
        >>> config_dict.key
        'value'
        """
        if name in self._dict:
            value = self._dict[name]
            logger.debug(f"Accessed attribute '{name}' with value: {value}")
            if isinstance(value, dict):
                return ConfigDict(value, root_config=self._root_config)
            return value
        logger.error(f"AttributeError: 'ConfigDict' object has no attribute '{name}'")
        raise AttributeError(f"'ConfigDict' object has no attribute '{name}'")

    def __setattr__(self, name:str, value):
        """
        Set a value in the dictionary and save the updated config.

        Parameters
        ----------
        name : str
            The key to set in the dictionary.
        value : Any
            The value to associate with the key.

        Examples
        --------
        >>> config_dict = ConfigDict({'key': 'value'})
        >>> config_dict.new_key = 'new_value'
        """
        if name in ('_dict', '_root_config'):
            super().__setattr__(name, value)
        else:
            self._dict[name] = value
            logger.debug(f"Set attribute '{name}' to '{value}'")
            if self._root_config:
                self._root_config._save_user_config()
                logger.info(f"User configuration saved after setting '{name}'")

    def __repr__(self):
        """
        Return the string representation of the dictionary.

        Returns
        -------
        str
            String representation of the dictionary.

        Examples
        --------
        >>> config_dict = ConfigDict({'key': 'value'})
        >>> repr(config_dict)
        "{'key': 'value'}"
        """
        return str(self._dict)

    def to_dict(self):
        """
        Returns the internal dictionary.

        Returns
        -------
        dict
            The internal dictionary.

        Examples
        --------
        >>> config_dict = ConfigDict({'key': 'value'})
        >>> config_dict.to_dict()
        {'key': 'value'}
        """
        logger.debug("Returning internal dictionary.")
        return self._dict


class Config:
    VARIABLE_PATTERN = re.compile(r'\{\{\s*(\w+)\s*\}\}|\$\(\s*(\w+)\s*\)')

    def __init__(self):
        """
        Initialize the Config object by loading the default and user-specific configurations.

        Examples
        --------
        >>> config = Config()
        """
        self._config = {}
        self._load_default_config()
        self._load_user_config()
        self._process_config_variables()
        logger.info("Config initialized with default and user-specific configurations")

    def __repr__(self):
        """
        Return the string representation of the config dictionary.

        Returns
        -------
        str
            String representation of the config dictionary.

        Examples
        --------
        >>> config = Config()
        >>> repr(config)
        "{'db_name': 'MatGraphDB', 'n_cores': 4}"
        """
        return str(self._config)

    def __getattr__(self, name:str):
        """
        Get a configuration value by name.

        Parameters
        ----------
        name : str
            The key to look for in the configuration dictionary.

        Returns
        -------
        Any
            The value associated with the key.

        Examples
        --------
        >>> config = Config()
        >>> config.db_name
        'MatGraphDB'
        """
        if name in self._config:
            value = self._config[name]
            logger.debug(f"Accessed configuration '{name}' with value: {value}")
            if isinstance(value, dict):
                return ConfigDict(value, root_config=self)
            return value
        logger.error(f"AttributeError: 'Config' object has no attribute '{name}'")
        raise AttributeError(f"'Config' object has no attribute '{name}'")

    def __setattr__(self, name:str, value):
        """
        Set a configuration value and save the updated config.

        Parameters
        ----------
        name : str
            The key to set in the configuration.
        value : Any
            The value to associate with the key.

        Examples
        --------
        >>> config = Config()
        >>> config.db_name = 'NewDB'
        """
        if name in ('_config', '_root_config', '_save_user_config') or name.startswith('_'):
            super().__setattr__(name, value)
        else:
            self._config[name] = value
            logger.debug(f"Set configuration '{name}' to '{value}'")
            self._save_user_config()

    def list_config(self):
        """
        List all configuration keys.

        Returns
        -------
        list
            A list of all configuration keys.

        Examples
        --------
        >>> config = Config()
        >>> config.list_config()
        ['db_name', 'n_cores', 'neo4j', 'logging_config']
        """
        logger.debug("Listing all configuration keys.")
        return list(self._config.keys())

    def get_config(self):
        """
        Get the entire configuration dictionary.

        Returns
        -------
        dict
            The entire configuration dictionary.

        Examples
        --------
        >>> config = Config()
        >>> config.get_config()
        {'db_name': 'MatGraphDB', 'n_cores': 4, 'neo4j': {...}, ...}
        """
        logger.debug("Returning the entire configuration dictionary.")
        return self._config

    def get(self, key:str, default=None):
        """
        Get a configuration value by key with an optional default.

        Parameters
        ----------
        key : str
            The key to look for in the configuration.
        default : Any, optional
            The default value to return if the key is not found, by default None.

        Returns
        -------
        Any
            The value associated with the key, or the default if not found.

        Examples
        --------
        >>> config = Config()
        >>> config.get('db_name')
        'MatGraphDB'
        >>> config.get('non_existent_key', 'default_value')
        'default_value'
        """
        value = self._config.get(key, default)
        logger.debug(f"Accessed configuration key '{key}' with value: {value}")
        return value

    def set_config(self, config_dict:dict):
        """
        Set multiple configuration values.

        Parameters
        ----------
        config_dict : dict
            A dictionary of configuration keys and values.

        Examples
        --------
        >>> config = Config()
        >>> config.set_config({'db_name': 'NewDB', 'n_cores': 8})
        """
        if not isinstance(config_dict, dict):
            logger.error("TypeError: Config must be a dictionary.")
            raise TypeError("Config must be a dictionary.")
        self._config.update(config_dict)
        logger.info(f"Updated configuration with: {config_dict}")
        self._save_user_config()

    def set(self, key:str, value):
        """
        Set a single configuration value.

        Parameters
        ----------
        key : str
            The key to set in the configuration.
        value : Any
            The value to associate with the key.

        Examples
        --------
        >>> config = Config()
        >>> config.set('db_name', 'NewDB')
        """
        self._config[key] = value
        logger.debug(f"Set configuration key '{key}' to value: {value}")
        self._save_user_config()

    def reset_to_default(self):
        """
        Reset the configuration to default settings.

        Examples
        --------
        >>> config = Config()
        >>> config.reset_to_default()
        """
        self._load_default_config()
        logger.info("Configuration reset to default settings.")
        self._save_user_config()

    def _load_default_config(self):
        """
        Load the default configuration from the package.

        Examples
        --------
        >>> config = Config()
        >>> config._load_default_config()
        """
        default_config_path = os.path.join(os.path.dirname(__file__), 'config.yml')
        with open(default_config_path, 'r') as f:
            self._config = yaml.safe_load(f) or {}
        logger.debug(f"Default configuration loaded from {default_config_path}")

    def _load_user_config(self):
        """
        Load user-specific configuration and update the default config.

        Examples
        --------
        >>> config = Config()
        >>> config._load_user_config()
        """
        user_config_paths = [
            os.path.join(os.getcwd(), 'config.yml'),            # Current directory
            os.path.expanduser('~/.matgraphdb/config.yml'),     # User's home directory
            os.environ.get('MATGRAPHDB_CONFIG')                 # Environment variable
        ]

        for path in user_config_paths:
            if path and os.path.exists(path):
                with open(path, 'r') as f:
                    user_config = yaml.safe_load(f) or {}
                    self._config.update(user_config)
                logger.info(f"User-specific configuration loaded from {path}")

    def _save_user_config(self):
        """
        Save the updated configuration to a user-specific file.

        Examples
        --------
        >>> config = Config()
        >>> config._save_user_config()
        """
        user_config_path = os.path.expanduser('~/.matgraphdb/config.yml')
        os.makedirs(os.path.dirname(user_config_path), exist_ok=True)
        with open(user_config_path, 'w') as f:
            yaml.dump(self._config, f)
        logger.info(f"User configuration saved to {user_config_path}")       # Environment variable
       
    def _process_config_variables(self):
        """
        Process and replace variables defined within {{ }} and $() in the configuration.

        This method iteratively processes the configuration dictionary, resolving variables up to a maximum of 10 iterations 
        to avoid infinite loops. Variables are expected to be in the format {{ variable_name }} or $(variable_name).

        Returns
        -------
        None

        Example
        -------
        >>> self._process_config_variables()
        """
        logger.debug("Starting variable substitution in configuration.")
        max_iterations = 10  # Prevent infinite loops
        for _ in range(max_iterations):
            old_config = str(self._config)
            self._config = self._resolve_variables(self._config)
            if str(self._config) == old_config:
                logger.debug("No further substitutions to be made, exiting loop.")
                break  # No more substitutions can be made
        else:
            logger.warning("Maximum iterations reached during variable substitution.")
        logger.debug("Variable substitution completed.")

    def _resolve_variables(self, config, parents=None):
        """
        Recursively resolve variables in the provided configuration.

        Parameters
        ----------
        config : dict, list, or str
            The configuration object in which variables will be resolved. This could be a dictionary, list, or string.
        parents : dict, optional
            The parent context containing previously defined variables, used for substitution (default is None).

        Returns
        -------
        dict, list, or str
            The configuration object with resolved variables.

        Example
        -------
        >>> resolved_config = self._resolve_variables(config)
        """
        if parents is None:
            parents = {}

        logger.debug(f"Resolving variables in config: {config}")
        
        if isinstance(config, dict):
            resolved = {}
            for key, value in config.items():
                resolved[key] = self._resolve_variables(value, {**parents, **config})
            return resolved
        elif isinstance(config, list):
            return [self._resolve_variables(item, parents) for item in config]
        elif isinstance(config, str):
            return self._substitute_variables(config, parents)
        else:
            return config

    def _substitute_variables(self, value, context):
        """
        Substitute variables in a given string based on the provided context.

        Parameters
        ----------
        value : str
            The string in which to substitute variables.
        context : dict
            The context containing variable names and their values for substitution.

        Returns
        -------
        str
            The string with substituted variables.

        Example
        -------
        >>> substituted_value = self._substitute_variables("Hello, {{name}}!", {"name": "World"})
        """

        logger.debug(f"Substituting variables in value: {value}")

        def replacer(match):
            var_name = match.group(1) or match.group(2)
            if var_name in context:
                replacement = context[var_name]
                logger.debug(f"Replacing variable '{var_name}' with '{replacement}'")
                return str(replacement)
            else:
                logger.warning(f"Variable '{var_name}' not found in context; leaving as is.")
                return match.group(0)  # Return the original string if variable not found

        return self.VARIABLE_PATTERN.sub(replacer, value)



FILE = Path(__file__).resolve()
PKG_DIR = str(FILE.parents[1])

config = Config()
config.set('pkg_dir', PKG_DIR)

np.set_printoptions(**config.numpy_config.np_printoptions.to_dict())

# if __name__ == "__main__":
    # config = Config()
    # print(config)
    # print(config.get('db_name'))
    # print(config.get('n_cores'))
    # print(config.get_config())

    # print(config.logging_config)
    # print(config.logging_config.version)

    # print(config.log_dir)
    # print(config.logging_config.handlers.file.filename)


    # os.makedirs(config.log_dir, exist_ok=True)