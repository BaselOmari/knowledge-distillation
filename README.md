# Knowledge Distillation

Reproducing “Distilling the Knowledge in a Neural Network” by Hinton et al.

## Relevant Files
- `distillation.py`
    - Creates a Small Model and trains it with knowledge distillation from a cumbersome model
- `distillation_no_three.py`
    - Creates a Small Model and trains it with knowledge distillation from a cumbersome model
    - Small model trained with digit 3 ommitted from the transfer set
    - After training, the learned bias for digit 3 is increased by +3.5
- `train.py`
    - Regular training script
    - Can be used to train cumbersome model or any regular model
- `layers.py`
    - Defines Cumbersome and Small Network models
    - Includes added functionality to scale and init weights
- `dataset.py`
    - MNIST dataset and dataloader
    - Includes filter used to remove digit 3 from dataset
- `training_params.py`
    - Includes training hyperparameters
    - Two setups are included:
        - `PaperParams`: Parameters used in Hinton et al.
        - `LocalParams`: Reduced parameters used for local demonstration

## Models Included
List of pretrained models included in `/models`:
- `cumbersome_model_1200.pt`
    - Cumbersome model with 1200 units in each hidden layer
    - Heavily regularized
- `distilled_model_800_20.pt`
    - Model with 800 units in each hidden layer
    - Trained with Knowledge Distillation from cumbersome model and with T = 20
- `small_model_800.pt`
    - Model with 800 units in each hidden layer
    - Trained without knowledge distillation or regularization
- `distilled_model_wo3_300_8.pt`
    - Model with 300 units in each hidden layer
    - Trained with Knowledge Distillation from cumbersome model and with T = 8
    - Trained with digit 3 ommitted from the transfer set
- `distilled_model_300_8.pt`
    - Model with 300 units in each hidden layer
    - Trained with Knowledge Distillation from cumbersome model and with T = 8

## How to Setup
1. Install Requirements: `pip install -r requirements.txt`
2. Run anyone of the following scripts with `python <script-name.py>`
    - `distillation.py`
    - `distillation_no_three.py`
    - `train.py`

## References
- G. Hinton, O. Vinyals and J. Dean, "Distilling the Knowledge in a Neural Network," in NIPS Deep Learning and Representation Learning Workshop, Montreal, 2015.
- G. Hinton, N. Srivastava, A. Krizhevsky, I. Sutskever and R. R. Salakhutdinov, "Improving neural networks by preventing co-adaptation of feature detectors," in arXiv preprint, 2012. 
- [WandB: How to Initialize Weights in PyTorch](https://wandb.ai/wandb_fc/tips/reports/How-to-Initialize-Weights-in-PyTorch--VmlldzoxNjcwOTg1)