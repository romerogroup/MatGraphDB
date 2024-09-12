from collections import defaultdict
import time


main_dict={}
def dict_iteration(n):
    d = {i: [i] for i in range(30)}
    processed_data = defaultdict(list)
    
    for i in range(n):
        for key, value in d.items():
            processed_data[key].append(value)
    return main_dict

def list_appending(n):
    l = []
    for i in range(n):
        l.append(i)
    return sum(l)

def benchmark(func, n):
    start = time.time()
    func(n)
    end = time.time()
    return end - start

# Run benchmarks
n = 1000000  # Number of elements
dict_time = benchmark(dict_iteration, n)
list_time = benchmark(list_appending, n)

print(f"Time to iterate through a dictionary with {n} elements: {dict_time:.6f} seconds")
print(f"Time to append to a list {n} times: {list_time:.6f} seconds")
print(f"Ratio (dict/list): {dict_time/list_time:.2f}")