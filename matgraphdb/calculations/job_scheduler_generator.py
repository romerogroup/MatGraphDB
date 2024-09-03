class JobSchedulerGenerator:
    def init_header(self):
        raise NotImplementedError("init_header must be implemented in the child class")
    def finalize(self):
        raise NotImplementedError("finalize must be implemented in the child class")

class SlurmScriptGenerator():

    def __init__(self, 
                job_name='mp_database_job', 
                partition='comm_small_day', 
                time='24:00:00'):
        
        self.job_name=job_name

        self.partition=partition
        self.time=time


        self.add_comp_resources=False

        self.slurm_header=''
        self.slurm_body=''
        self.slurm_script=''

    def add_slurm_header_argument(self, argument:str):
        if self.slurm_header=='':
            raise ValueError("Slurm header is empty. Call init_slurm_header first")
        self.slurm_header+= argument + '\n'
        
    def add_slurm_script_body(self, command:str):
        self.slurm_body+=command + '\n'
        return self.slurm_body
    
    def init_header(self):
        self.slurm_header+='#!/bin/bash\n'
        self.slurm_header+=f'#SBATCH -J {self.job_name}\n'
        self.slurm_header+=f'#SBATCH -p {self.partition}\n'
        self.slurm_header+=f'#SBATCH -t {self.time}\n'
        return self.slurm_header

    def add_slurm_header_comp_resources(self, n_nodes=None, n_tasks=None, cpus_per_task=None):
        self.add_comp_resources=True
        self.n_tasks=n_tasks
        self.cpus_per_task=cpus_per_task
        self.n_nodes=n_nodes
        command=''
        if n_nodes is not None:
            command+=f'#SBATCH --nodes={self.n_nodes}\n'
        if n_tasks is not None:
            command+=f'#SBATCH -n {self.n_tasks}\n'
        if cpus_per_task is not None:
            command+=f'#SBATCH -c {self.cpus_per_task}\n'

        self.slurm_header+=command
        return self.slurm_header
    
    def exclude_nodes(self, node_names=[]):
        if self.slurm_header=='':
            raise ValueError("Slurm header is empty. Call init_slurm_header first")
        node_list_string= ','.join(node_names)
        command=f'#SBATCH --exclude={node_list_string}\n'
        self.slurm_header+=command
        return command

    def finalize(self):
        if self.slurm_header=='':
            raise ValueError("Slurm header is empty. Call init_slurm_header first")
        if not self.add_comp_resources:
            raise ValueError("Add computational resources before finalizing. add_slurm_header_comp_resources")
        if self.slurm_body=='':
            raise ValueError("Slurm body is empty. Call add_slurm_script first")
        
        self.slurm_script=self.slurm_header + '\n' + self.slurm_body
        return self.slurm_script
    