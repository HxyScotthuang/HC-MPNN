# Link Prediction with Relational Hypergraphs

This is the official code base for the paper **Link Prediction with Relational Hypergraphs**. (TMLR 2025/05)

## Installation
You can install the dependencies with pip (or conda), and it works with python 3.9+, pytorch 2.1.0, and pytorch-geometric 2.3.0, tqdm.
```
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install torch-scatter torch-sparse torch-geometric -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install tqdm
```

## Reproducing the experiments
To reproduce the experiments of HCNet, you need to specify the corresponding arguments shown in the src/config.py file. For example, if there is GPU support, then
```
python main.py 
```
will automatically run the best configuration of FB-AUTO, but you can also specify runs of other datasets. 

An example of running WP-IND with GPU support would be 
```
python main.py --dataset WP-IND --lr 5e-3 --num_layer 5 --neg_ratio 10 
```
and the one without GPU support would be
```
python main.py --dataset WP-IND --lr 5e-3 --num_layer 5 --neg_ratio 10 --gpu -1
```


## Using the powerful triton kernel

To scale the model up, we additional implement a powerful Triton kernel for message passing on relational hypergraphs, which you can use now by setting flag ``` --use_triton ```.
```
python main.py --dataset WP-IND --lr 5e-3 --num_layer 5 --neg_ratio 10 --use_triton
```

This will approximately double the speed, and greatly reduce the space complexity from $O(Ed)$ to $O(Vd)$ during the message passing, as we never materialized the messages! 

However, to let this work, you need some additional upgrade for both pytorch_geometric and triton.
```
pip install git+https://github.com/pyg-team/pytorch_geometric.git
pip install -U --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly
```

## Synthetic Experiment

We have included the synthetic experiment of *HyperCycle* described in the appendix of the paper in **Synthetic.ipynb**. 

## 📖 Citation

If you find this work useful, please consider citing our TMLR 2025 paper:

```bibtex
@article{huang2025link,
  title   = {Link Prediction with Relational Hypergraphs},
  author  = {Xingyue Huang and Miguel Romero Orth and Pablo Barcel{\'o} and Michael M. Bronstein and {\.{I}}smail {\.{I}}lkan Ceylan},
  journal = {Transactions on Machine Learning Research},
  issn    = {2835-8856},
  year    = {2025}
}

