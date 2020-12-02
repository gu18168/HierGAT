# HierGAT

This is the implementation of "Entity Resolution via Hierarchical Graph Attention Network"

## Environment

* pytorch 1.3.1
* python 3.6.10

## Datasets

More datasets can be found at https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md

## Data Pre-Process

We first need to transform the raw data set into a graph structure. An example of an Amazon-Google dataset:

```
python preprocess.py -d dataset/Amazon-Google/ -i Amazon.csv -i Google.csv -o output -m Mapping.csv -p "id:id;brand:manu,title,price;cate:desc" -mo dataset/wiki.en.bin
```

## Train HierGAT

```
python train.py -d output/
```