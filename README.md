# Link Prediction with Relational Hypergraphs

This is the official code base for the paper *Link Prediction with Relational Hypergraphs*.

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
