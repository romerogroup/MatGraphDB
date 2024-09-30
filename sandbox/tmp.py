import inspect
import pandas as pd


def get_function_args(func):
    signature = inspect.signature(func)
    params = signature.parameters
    
    args = []
    kwargs = []
    
    for name, param in params.items():
        if param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
            if param.default == inspect.Parameter.empty:
                args.append(name)
            else:
                kwargs.append(name)
        elif param.kind == inspect.Parameter.KEYWORD_ONLY:
            kwargs.append(name)
    
    return args, kwargs


df=pd.DataFrame({'a':[1,2,3],'b':[4,5,6], 'x':[4,5,6]})

for i,row in df.iterrows():
    # print(row)
    row_data=row.drop('x').to_dict()
    print(row_data)

def calc_func(row_data: dict, calc_dir: str, a=0):
    pass

# calc=inspect.signature(calc_func)

calc=get_function_args(calc_func)

print(calc)