import os
import logging
from typing import Union, List, Dict
import importlib

from parquetdb import ParquetDB

logger = logging.getLogger(__name__)

def load_store(store_path: str, default_store_class = None):
    store_metadata=ParquetDB(store_path).get_metadata()
    class_module = store_metadata.get('class_module', None)
    class_name = store_metadata.get('class', None)
    
    logger.debug(f"Class module: {class_module}")
    logger.debug(f"Class: {class_name}")
    
    if class_module and class_name and default_store_class is not None:
        logger.debug(f"Importing class from module: {class_module}")
        module = importlib.import_module(class_module)
        class_obj = getattr(module, class_name)
        store = class_obj(storage_path=store_path)
    else:
        logger.debug(f"Using default store class: {default_store_class.__name__}")
        store = default_store_class(storage_path=store_path)
        
    return store
        
            