import time
import statistics

def time_function(func, *args, iterations=1000000):
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func(*args)
        end = time.perf_counter()
        times.append(end - start)
    return statistics.mean(times)

def try_except_approach(dict_obj, key):
    try:
        return dict_obj[key]
    except KeyError:
        return None

def if_else_approach(dict_obj, key):
    if key in dict_obj:
        return dict_obj[key]
    else:
        return None

# Test scenarios
test_dict = {'a': 1, 'b': 2, 'c': 3}

# Scenario 1: Key exists
key_exists = 'b'

# Scenario 2: Key doesn't exist
key_not_exists = 'd'

# Run tests
try_except_time_exists = time_function(try_except_approach, test_dict, key_exists)
if_else_time_exists = time_function(if_else_approach, test_dict, key_exists)

try_except_time_not_exists = time_function(try_except_approach, test_dict, key_not_exists)
if_else_time_not_exists = time_function(if_else_approach, test_dict, key_not_exists)

# Print results
print(f"Try-Except (key exists): {try_except_time_exists:.9f} seconds")
print(f"If-Else (key exists): {if_else_time_exists:.9f} seconds")
print(f"Try-Except (key doesn't exist): {try_except_time_not_exists:.9f} seconds")
print(f"If-Else (key doesn't exist): {if_else_time_not_exists:.9f} seconds")