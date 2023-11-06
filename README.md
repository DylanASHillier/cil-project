# cil-project
Continuous Learning Competition Project
## Contribution Guidelines
- Please work on branches during the day, and make PRs to merge into main
- All merges should be approved by one other person, you can request or not
- Either approve PR or ask for revisions. Prune inactive prs
- Do not add training data to the repo, this should always be specified by path
- Be perfect so we don't need tests...

## Installation Instructions
From the root directory:
- Install python >= 3.10
- Run `pip install -r requirements.txt`
- Run `pip install -e .`

From colab via github:
See [this colab](https://colab.research.google.com/drive/1eCuiuP1UD7NzRYIDavccUpu01y1Mgjl2?usp=sharing) for how to install the package from github
- Run `!pip install git+https://github.com/DylanASHillier/cil-project.git`

## Usage Instructions
Please use linting and formatting tools to make
sure your code is clean... I am using `black` and
`pylint`.

In particular you can lint like `pylint --rcfile=pylintrc $(git ls-files '*.py')` once you install the package.

```
python cilproject/experiment.py "./data" --pred-path="preds/results" --common-model=id --memory-type=kmeans --phase-model="timm_classifier_residual" --aggregation="add" --embedding-model=vit_base_patch16_224.augreg2_in21k_ft_in1k --num-epochs=10 --classifier-name=linear --random-seed=42 --save-preds --disable-val
```