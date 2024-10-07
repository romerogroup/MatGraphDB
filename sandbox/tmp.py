import os
import inspect
import pandas as pd
import shutil
import re
import logging
import yaml
import logging.config

from matgraphdb import MatGraphDB, config, logging_config


# # # config.reset_to_default()

# print(logging_config.loggers)
# logging_config.loggers.matgraphdb.level='DEBUG'
# print(logging_config.loggers)


print(config.logging_config.loggers)
logging_config.loggers.matgraphdb.level='DEBUG'
print(config.logging_config.loggers)

# logging.config.dictConfig(logging_config.to_dict())
# logging.config.dictConfig(logging_config.to_dict())

# # logger = logging.getLogger('matgraphdb')
# # print(f"Logger level: {logger.getEffectiveLevel()}")

database_dir=os.path.join(config.data_dir,'MatGraphDB')
mgdb=MatGraphDB(database_dir)
shutil.rmtree(database_dir)

# config.logging_config.loggers.matgraphdb.level='DEBUG'

mgdb=MatGraphDB(database_dir)
# mgdb.config.set('db_name','MatGraphDB')


# class Foo:
#     def __init__(self, a, b):
#         self.a = a
#         self.b = b

# class ConfigDict(dict):
#     def __init__(self, dictionary: dict ):
#         dictionary=substitute_template_keys(dictionary)
#         for key, value in dictionary.items():
#             if isinstance(value, dict):
#                 self.__dict__.update({key: ConfigDict(value)})
#             else:
#                 self.__dict__.update({key: value})

#     def __getitem__(self, key):
#         return getattr(self, key)

#     def __setitem__(self, key, value):
#         if isinstance(value, dict) or isinstance(value, ConfigDict):
#             value = getattr(self, key)
#             value.update(value)
        
#         setattr(self, key, value)

#     def __repr__(self):
#         return __class__.__name__ + '(' + repr(self.__dict__) + ')'

#     def update(self, dictionary):
#         for key, value in dictionary.items():
#             if isinstance(value, dict):
#                 self.__dict__.update({key: ConfigDict(value)})
#             else:
#                 self.__dict__.update({key: value})

#     def to_dict(self, dict=None):
#         dictionary={}
#         for key, value in self.__dict__.items():
#             if isinstance(value, ConfigDict):
#                 dictionary.update({key: value.to_dict()})
#             else:
#                 dictionary.update({key: value})
#         return dictionary

#     @classmethod
#     def from_yaml(cls, yaml_file):
#         with open(yaml_file, 'r') as f:
#             config = yaml.safe_load(f)
#         return cls(config)

#     @classmethod
#     def from_json(cls, json_file):
#         with open(json_file, 'r') as f:
#             config = json.load(f)
#         return cls(config)
# def substitute_template_keys(d):
#     # First, we need to gather all direct key-value pairs where keys are NOT in the form {{ var }}
#     plain_values = {}
    
#     # Function to collect plain values from the dictionary (non-template keys)
#     def collect_plain_values(d):
#         for key, value in d.items():
#             if isinstance(value, dict):
#                 collect_plain_values(value)
#             else:
#                 # Collect non-template values
#                 if not re.match(r'\{\{\s*\w+\s*\}\}', key):
#                     plain_values[key] = value

#     # Call to gather plain values for reference
#     collect_plain_values(d)

#     # Recursive function to substitute {{ var }} with its corresponding value
#     def substitute_in_dict(d):
#         new_dict = {}
#         for key, value in d.items():
#             # Check if the key matches the template '{{ var }}'
#             template_match = re.match(r'\{\{\s*(\w+)\s*\}\}', key)
#             if template_match:
#                 var_name = template_match.group(1)
#                 # Find the value of 'var_name' in the dictionary or plain_values
#                 if var_name in plain_values:
#                     substituted_key = plain_values[var_name]
#                     new_dict[substituted_key] = value
#             else:
#                 new_dict[key] = value
            
#             # If value is a dictionary, recursively substitute in it
#             if isinstance(value, dict):
#                 new_dict[key] = substitute_in_dict(value)
#         return new_dict

#     # Perform the substitution
#     return substitute_in_dict(d)


# a = {'a':{'b':{'{{ c }}':5}, 'c': 'reeeeeee'}, '{{ c }}':2}
# updated_dict = substitute_template_keys(a)
# print(updated_dict)

# a = {'a':{'b':{'{{ c }}':5}, 'c': 'reeeeeee'}, '{{ c }}':2}
# # result=find_template_keys(a)
# # print(result)
# config= ConfigDict(a)


# print(config.to_dict())
# # print(config['a'].b['c'])
# # config['a'].update({'c':{'c':6}})
# # # config.a=1
# # print(config.__dict__)
# print(config.a.b.reeeeeee)
# # config.a.b.c=1
# # print(config.a.b.c)