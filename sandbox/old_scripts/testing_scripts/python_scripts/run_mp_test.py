import os


def get_cpus_per_node():
    cpus_per_node = os.getenv('SLURM_JOB_CPUS_PER_NODE')
    if '(x' in cpus_per_node:
        cpu_per_node, num_nodes= cpus_per_node.strip(')').split('(x')
        cpus_node_list = [int(cpu_per_node) for _ in range(int(num_nodes))]
    else:
        cpus_node_list = [int(x) for x in cpus_per_node.split(',')]
    return cpus_node_list

def get_num_tasks():
    num_tasks = os.getenv('SLURM_NTASKS')
    if num_tasks:
        num_tasks = int(num_tasks)
    return num_tasks

def get_num_nodes():
    num_nodes = int(os.getenv('SLURM_JOB_NUM_NODES'))
    return num_nodes

def get_total_cores(cpus_per_node):
    return sum(cpus_per_node)

def main():

    print(os.getenv('SLURM_JOB_CPUS_PER_NODE'))
    print(os.getenv('SLURM_NTASKS'))
    print(os.getenv('SLURM_JOB_NUM_NODES'))

    cpus_per_node = get_cpus_per_node()
    # Get the number of CPUs allocated per node (as a string)
    print(type(cpus_per_node))
    print(cpus_per_node)

    num_tasks = get_num_tasks()
    print(type(num_tasks))
    print(num_tasks)

    num_nodes = get_num_nodes()
    print(type(num_nodes))
    print(num_nodes)

    total_cores = get_total_cores(cpus_per_node)
    print(type(total_cores))
    print(total_cores)






if __name__ == '__main__':
    main()