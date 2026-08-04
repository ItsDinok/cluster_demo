# Cluster Demonstration

This will serve as a demonstration of how to use the cluster for simple machine learning tasks.

## 1. Basic Information

Basic information about how or why to use the cluster can be found in [Villanelle's workshop](https://github.com/LCAS/novel_hpc_workshop/blob/main/readme.md?plain=1). This provides a good template to use, but this will explain why parameters are chosen or attempt to address some of the more common pitfalls one may encounter when using slurm.

## 2. Setting up the environment

The cluster has some quirks that makes some tooling difficult outside of docker containers. While it may be tempting to use docker containers, I found setting them up to be frustrating and tedious, with minimal room for LLMs to help. This will be expanded upon in a federated learning on the cluster guide. Below is some boilerplate commands to get it up and running. Please note you need to be on the university network in order for it to work, this can be done via the Cisco VPN or a VPN of your choice, I will not cover setting up SSH keys as this has been done already by Villanelle and can also be assisted with an LLM of your choice.

```console
# Log into the server
ssh <first initial><surname>@login.novel.hpc

# Install utilities
mkdir -p ~/.local/bin
cd ~/.local/bin
for cmd in lsgpu lsjob erun epy epip; do
    curl -sL "https://s.vnet.tel/$cmd" -o "$cmd"
    chmod +x "$cmd"
done
source ~/.bashrc
```

The second half of this will give you walls of python code at first. It is normal and will conclude. This command gives you access to some powerful commands:
- `lsgpu`: Shows you a table of GPUs and their available resources.
- 'lsjob': Shows you a table of currently running tasks and their statuses. The `-w` flag can be appended to it to make it real-time
- `erun`: Requests a GPU and loads you into a bash shell inside a pytorch enroot container
- `epip` `epy`: Allows you to safely run python or pip inside a venv on a GPU compute node

### Starting a task

The following commands will start and run the code from this demo:

```console
git clone https://github.com/itsdinok/cluster_demo
cd cluster_demo

# Inside this you can create a virtual environment using a compute node
epy -m venv --copies --system-site-packages .venv
source .venv/bin/activate

# Install requirements
epip install -r requirements.txt

# Run it
sbatch run.sh
```

NEVER do a standard `./run.sh` or `python mnist.py` as this will run it on the resource-constrained compute nodes and make the administrators very unhappy.

`srun` can be used to run an interactive job, but tasks on the cluster should be engineered to avoid the need for this. `sinfo` and `squeue` will show information regarding active tasks. `scancel <jobid>` will cancel a specific job. To my knowledge you can only cancel your own jobs but be careful as there is no validation, if you enter the wrong index you will cancel the wrong job and may undo hours of waiting.

I like to follow log files using `tail -f logs/task.log`, which, if configured properly, will give a running log of your file's output. Learning how to use this will be in the next section.

## 3. Building a task

Machine vision or big data tasks can be bottlenecked by data loading, which can diminish but not eliminate the advantages of using a cluster, for this reason a suitable dataloader should be written, this can be found at the end of this document. Below is a breakdown of the shell file we used for this, the full code can be found in `run.sh`.

```bash
#SBATCH --job-name=mnist_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
```

These are standard headers for a run.sh, they name the task, ask to use GPUs, ask for 64GiB of memory, and ask for 8 CPUs. I do not know why 8 CPUs is the standard, but it seems to be and as there is no competition for CPU space there is no reason to change it. You can request more memory, I have used 96 and 128GiB for ViT tasks, but generally ask for only what you need as it is a shared resource. The name isn't terribly important if you only run one task at a time as your name will be displayed alongside your task, but do set specific names if you run multiple jobs at once.

```bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your@email.address
```

This allows you to get email updates regarding your jobs. This is a pretty useful feature if your tasks are going to take hours. There is no validation so it can be removed or a placeholder email can be used.

```bash
#SBATCH --time=8:00:00
```

This is the default time command when using the normal queue. In the normal queue you cannot exceed 24 hours. If you need to run it for longer use the alternative sequence:

```bash
#SBATCH --qos=long
#SBATCH --time=2-00:00:00
```

which requests two days. The maximum with long QoS is seven days.

```bash
#SBATCH --output=logs/mnist_%j.log
#SBATCH --error=logs/mnist_%j.err
```

This handles the logging. Any error will be written to the .err file, any output will be put in the .log file. From experience it is a good idea to routinely clean your log directory if you are running the same test multiple times. Also design pretty printouts for your code so you don't have to trawl through a long wall of text as log files record ALL output.

```bash
#SBATCH --nodelist=hpc-novel-gpu[01-06]
#SBATCH --nodes=1
#SBATCH --ntasks=1
```

This asks for GPUs from a specific node, in this case any from 01-06. This may be beneficial if you really want to use a specific architecture, you can see which GPUs are on which nodes with `lsgpu`. It should be noted that some GPU architectures make PyTorch substantially more awkward to use. I would stay within 01-06 unless you have a need for a different architecture or model. Multinode and multitask may be possible with this cluster as the flags are there, but I am yet to find a purpose for them and as such I can't say with certainty that they _do_ work. 

```bash
#SBATCH --container-image=/home/shared/air/enroot-images/pytorch_n_friends.sqsh
#SBATCH --contaienr-mount-home
#SBATCH --container-mounts=/home/shared:/home/shared:ro
```

This is enroot configurations. I would not recommend tweaking these unless you absolutely need to.

```bash
module purge
cd $SLURM_SUBMIT_DIR
echo "Starting task $PWD"
srun bash -c "cd $SLURM_SUBMIT_DIR && python mnist.py"
```

This is the last part that clears potentially interfering modules, navigates to the working directory, and runs the task. At this point you can run your job!

## 4. Decent housekeeping

I would recommend using `sperf <jobid>` to monitor your memory and CPU every now and again, you may be able to revise your requests down in subsequent tests and free up more usage for everyone else. If everyone asks for only what they need there will be less queues for jobs! 

You have a decent amount of storage on the cluster, and some people use VSCode to develop on the cluster. Sometimes developing and tweaking on the cluster is necessary as it is much faster than pulling from git. To this end I use vi or vim, but you can install anything that lives in userland. 

Logs can be monitored with `tail -f logs/*`, but this requires decently attentive clearing of logfiles. 

Any utilities built for working with the cluster should be shared in the wiki and will be appreciated!

**BUILD HPC GUARDRAILS INTO YOUR JOBS TO PREVENT THEM FROM RUNNING ON LOGIN NODES PLEASE**

```python
# Simple guardrail
if not torch.cuda.is_available():
    print("You are attempting to run a GPU-bound script but python cannot find a GPU. You likely submitted a job without requesting a GPU.")
    sys.exit(1)
```

## 5. Dataloaders

For batch-based machine learning you can find cyclical and prolonged slowdowns come from using disk I/O and CPU for data loading during training. Eight CPUs can somewhat reduce this, but it is still substantially slower than using GPU-based loading. The following code snippet is a high-speed data loader for MNIST, you will have to make your own for other datasets. There will still be a substantial performance increase over local training without a custom loader, so don't stress if it is complex.

```python

class FastMNISTLoader(Dataset):
    """
    Zero Disk I/O. Zero CPU augmentations during training. Pure GPU feeding.
    """
    def __init__(self, root, train):
        filename = f"fast_mnist_{'train' if train else 'val'}.npz"
        filepath = os.path.join(root, filename)
        
        if not os.path.exists(filepath):
            print(f"❌ Cannot find {filepath}!")
            print("You must run 'generate_fast_data.py' (and set the right dataset directory) first.")
            sys.exit(1)
            
        print(f"--> [SUPERFAST] Loading pre-computed arrays from {filepath}...")
        npz_file = np.load(filepath)
        
        # Convert numpy arrays to PyTorch tensors
        self.data = torch.from_numpy(npz_file['data'])
        self.targets = torch.from_numpy(npz_file['targets']).long()
        self.train = train
        self.num_variants = self.data.shape[1] # e.g., 6 variants for train
        
        # Array holding which variant to use for each image (updated per epoch)
        self.current_choices = torch.zeros(len(self.data), dtype=torch.long)
        print(f"--> [SUPERFAST] Loaded {self.num_variants} variant(s) per image into RAM.")

    def set_epoch(self):
        """Randomly pick 1 of the N pre-computed variants for this epoch."""
        if self.train:
            self.current_choices = torch.randint(0, self.num_variants, (len(self.data),))

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        # O(1) Instant Memory Lookup: CPU does zero math here.
        chosen_variant_idx = self.current_choices[idx]
        return self.data[idx, chosen_variant_idx], self.targets[idx]
```

You may have noticed that it uses fast\_mnist.epz files, these are generated by running `generate_fast_data.py`, credit to Villanelle for this one. 

## 6. Closing 

This is a breakdown of how to get started with the cluster, but as you do more and more complex tasks you will invariably find that you need to come up with creative solutions to problems you couldn't comprehend before starting this. Please document them :) 
