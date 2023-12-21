# Columbia IEORE4530: Fair Ranking Project

This repository contains the code used for the experiments in Fair Ranking porject for Fall 2023 IEORE4530 course at Columbia University. The contributors for this projects are Poch Laohrenu (pl2839@columbia.edu), Jirat Suchato (js5808@columbia.edu), Leo Hu (ch3729@columbia.edu), and Richard Uzeel (ru2155@columbia.edu). The repository is largely based on the repo of the progonal paper "Fair Ranking as Fair Division: Impact-Based Individual Fairness in Ranking" ([KDD2022](https://kdd.org/kdd2022/)) by [Yuta Saito](https://usait0.com/en/) and [Thorsten Joachims](https://www.cs.cornell.edu/people/tj/).


## Requirements and Setup

```bash
# clone the repository
git clone https://github.com/pochl/AGM2023-FairRanking.git
```

Create virtual environment with conda
```
conda create -n agm python=3.9
conda activate agm
```

Install required packages
```
pip install -r requirements.txt
```

Since `cvxpy` cannot be installed using pip on Macbook with M1 chip, it can be installed using the following comman. For other OS, this command should work too. 
```
conda install -c conda-forge cvxpy
```

## Running the Experiments

The experiemnts were done in the `src/experiment.ipynb` notebook. 


