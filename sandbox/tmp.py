import os
import inspect
import pandas as pd
import shutil
import re
import logging
import yaml
import logging.config

from matgraphdb import MatGraphDB, config, logging_config


def magicmethod(clazz, method):
    if method not in clazz.__dict__:  # Not defined in clazz : inherited
        return 'inherited'
    elif hasattr(super(clazz), method):  # Present in parent : overloaded
        return 'overloaded'
    else:  # Not present in parent : newly defined
        return 'newly defined'
    
    
# import inspect
# for cls in MyClass.mro():
#     if 'my_method' in cls.__dict__:
#         print(f"my_method is defined in {cls}")
#         print(inspect.getsource(cls.my_method))
#         break