# blood-cell-classificator
# Blood Cell Classification using MLNN and CNN

## Models and Tools

## Models Implemented

- **MLNN (Multi-Layer Neural Network)**  
  A fully connected feedforward network with multiple dense layers. Simpler and faster to train, suitable for baseline comparisons.

- **CNN (Convolutional Neural Network)**  
  A deeper architecture with convolutional and pooling layers, designed to capture spatial patterns. Achieved the best overall performance.

Each model was trained using different configurations:

- **Optimizers:**
  - `Adam`: Adaptive learning rate, fast convergence.
  - `SGD + Momentum`: More stable, better generalization in some cases.

- **Loss Functions:**
  - `Cross-Entropy`: Standard for classification tasks.
  - `Cross-Entropy with Class Weights`: Improves recall for minority classes.
  - `Negative Log-Likelihood (NLL)`: Alternative formulation, competitive in CNNs.

## Tools and Libraries

- Python 3.10+
- PyTorch
- NumPy
- Matplotlib / Seaborn
- Scikit-learn
- LaTeX (for report generation)
- Jupyter Notebooks (for exploratory analysis)

The report is availble [here](Docs/Report.pdf).

## Authors
MSc Student
- Pedro Trindade
