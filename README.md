# self-pruning-neural-network

This project implements a neural network that learns to prune itself during training using learnable gates and L1 regularization.

## Run

pip install -r requirements.txt  
python src/train.py


##Observation on Sparsity vs Lambda

Despite experimenting with different values of λ, the sparsity level remained low (~1.7%) across all runs. This indicates that the L1 regularization strength was insufficient to push the gate values toward zero.

Possible reasons:

The number of training epochs (5) is too low for gates to converge
The λ values are relatively small to enforce strong sparsity
The model prioritizes classification accuracy over pruning

##Conclusion

The experiment demonstrates that:

Simply adding L1 regularization is not always sufficient
Proper tuning of λ and longer training is required
There exists a trade-off between sparsity and accuracy, though not strongly observed here
