

# PMF-CPI：Assessing drug selectivity with a pretrained multi-functional model for compound-protein interactions

## Brief introduction

Here we present a sequence-only and pretrained multi-functional model for compound-protein interaction prediction (PMF-CPI) and fine-tune it to assess drug selectivity.  PMF-CPI can accurately predict different drug affinities or opposite interactions toward similar targets, recognizing selective drugs for precise therapeutics.  The code is implementation on PyTorch. 

In this repository, we provide datasets for different tasks of classification and regression. The overview of our CPI prediction  is as follows:

![framework](img/framework.png)

The detail of our model are described in our paper.



## Requirements

* PyTorch

* torch-geometric

* RDKit

We utlized three embedding methods as follows:

* TAPE: https://github.com/li-ziang/tape

* ESM-2:https://github.com/facebookresearch/esm

* Bio2Vec:https://github.com/kyu999/biovec

## Usage

Here we provide classification and regression 

### Training of our model using your drug selectivity dataset

### Fintune of our model using your drug selectivity dataset





## TODO

Note that protein sequence and SMILES sequence have various lengths, 







