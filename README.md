# self-pruning-neural-network

This project implements a neural network that learns to prune itself during training using learnable gates and L1 regularization.

## Run

pip install -r requirements.txt  
python src/train.py


##  Main Idea

Each weight in the network is associated with a **learnable gate parameter**:

- Gate values are constrained between 0 and 1 using a sigmoid function  
- Effective weight = original weight × gate  
- If gate → 0 ⇒ connection is pruned  

## Loss Function

The training objective combines classification loss with a sparsity penalty:

Total Loss = CrossEntropyLoss + λ × Sparsity Loss

- CrossEntropyLoss → ensures classification performance  
- L1 Sparsity Loss → encourages gates to become zero  

##  Architecture

- Custom `PrunableLinear` layer
- Feedforward Neural Network:
  - Input: CIFAR-10 images (32×32×3)
  - Hidden layers: 512 → 256 neurons
  - Output: 10 classes

##  Results

| Lambda (λ) | Test Accuracy (%) | Sparsity (%) |
|------------|------------------|--------------|
| 0.0001     | 41.90            | 1.69         |
| 0.001      | 42.49            | 1.71         |
| 0.01       | 42.20            | 1.70         |

---

##  Observations

- Sparsity remained low (~1.7%) across all λ values  
- Increasing λ did not significantly impact pruning  
- Accuracy remained stable across experiments  

###  Interpretation

This indicates that:
- The regularization strength (λ) was insufficient to enforce strong sparsity  
- The number of training epochs (5) was too low for gate convergence  
- The model prioritized classification performance over pruning  


##  Trade-off Insight

Although not strongly observed here, typically:

- Higher λ → Higher sparsity but lower accuracy  
- Lower λ → Better accuracy but minimal pruning  
