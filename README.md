# Fair Ranking as Fair Division: Impact-Based Individual Fairness in Ranking (KDD2022)

This repository contains the code used for the experiments in "Fair Ranking as Fair Division: Impact-Based Individual Fairness in Ranking" ([KDD2022](https://kdd.org/kdd2022/)) by [Yuta Saito](https://usait0.com/en/) and [Thorsten Joachims](https://www.cs.cornell.edu/people/tj/).

## Abstract

Rankings have become the primary interface of many two-sided markets. Many have noted that the rankings not only affect the satisfaction of the users (e.g., customers, listeners, employers, travelers), but that the position in the ranking allocates exposure -- and thus economic opportunity -- to the ranked items (e.g., articles, products, songs, job seekers, restaurants, hotels). This has raised questions of fairness to the items, and most existing works have addressed fairness by explicitly linking item exposure to item relevance. However, we argue that any particular choice of such a link function may be difficult to defend, and we show that the resulting rankings can still be unfair. To avoid these shortcomings, we develop a new axiomatic approach that is rooted in principles of *fair division*. This not only avoids the need to choose a link function, but also more meaningfully quantifies the impact on the items beyond exposure. Our axioms of *envy-freeness* and *dominance over uniform ranking* postulate that in fair rankings every item should prefer their own rank allocation over that of any other item, and that no item should be actively disadvantaged by the rankings. To compute ranking policies that are fair according to these axioms, we propose a new ranking objective related to the *Nash Social Welfare*. We show that the solution has guarantees regarding its envy-freeness, its dominance over uniform rankings for every item, and its Pareto optimality. In contrast, we show that conventional exposure-based fairness can produce large amounts of envy and have a highly disparate impact on the items. Beyond these theoretical results, we illustrate empirically how our framework controls the trade-off between impact-based individual item fairness and user utility.

## Citation

```
@inproceedings{saito2022fair,
  title={Fair Ranking as Fair Division: Impact-Based Individual Fairness in Ranking},
  author={Saito, Yuta and Joachims, Thorsten},
  booktitle = {Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages = {1514–1524},
  year={2022}
}
```

## Requirements and Setup

```bash
# clone the repository
git clone https://github.com/usaito/kdd2022-fair-ranking-nsw.git
cd src

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


We also use [`pyxclib`](https://github.com/kunaldahiya/pyxclib) to handle extreme classification data. This tool cannot be installed via `pip`, so please build this tool as follows.

```bash
git clone https://github.com/kunaldahiya/pyxclib.git
cd pyxclib

python setup.py install
```

## Datasets

We use "Delicious" and "Wiki10-31K" from [The Extreme Classification Repository](http://manikvarma.org/downloads/XC/XMLRepository.html). Please install the above datasets from the repository and put them under the `./src/real/` as follows.

```
root/
　├ synthetic/
　├ real/
　│　└ data/
　│　├── delicious.txt
　│　├── wiki_train.txt
　│　└── wiki_test.txt
```

Note that we rename
- `Delicious_data.txt` in Delicious.zip to `delicious.txt`
- `test.txt` in Wiki10.bow.zip to `wiki_test.txt`
- `train.txt` in Wiki10.bow.zip to `wiki_train.txt`
, respectively.

## Running the Code

The experimental workflow is implemented using [Hydra](https://github.com/facebookresearch/hydra).
The commands needed to reproduce the experiment of each section are summarized below. Please move under the `src` directly first and then run the commands. The experimental results (including the corresponding figures) will be stored in the `logs/` directory.

### Synthetic Data (Section 5.1 and Appendix)

```bash

# varying lambda (Figures 1 and 2)
python synthetic/main_lam.py

# varying noise level (Figure 3)
python synthetic/main_noise.py

# varying K (Figure 5)
python synthetic/main_k.py

# varying number of items (Figure 6)
python synthetic/main_n_doc.py

# varying number of items (Figure 7)
python synthetic/main_lam.py setting.exam_func=exp
```

### Real-World Data (Section 5.2 and Appendix)

```bash
run python real/main.py setting.dataset=d,w -m
```
