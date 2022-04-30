# Neural Message Passing for Quantum Chemistry (CMU 10-708 Final Project Spring 2022)

Baseline model found at: https://github.com/priba/nmp_qc.  
We thank the authors for providing an invaluable starting point for a few inexperienced but curious graduate students 


## Installation

    $ pip install -r requirements.txt
    $ python main.py

For help
    
    $ python main.py --help

The default behavior is to search for the best model and resume training, in order to do this the main.py script must be ran on GPUs. 

To begin training a model from scratch simply specify a model directory 

    $ python main.py --resume "path/to/folder"

# Data
The data used in this project can be downloaded [here](https://drive.google.com/file/d/11DR28EFbWWR_EKbVwr-RKMVeLwC2gYft/view?usp=sharing)

After cloning the repo, unzip the downloaded zip file above in the directory `./data/qm9/`.
