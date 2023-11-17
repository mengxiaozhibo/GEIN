This code is for reviewing of the paper titled 'Graph-Enhanced Interest Network for CTR Prediction', which has being submitted to the International World Wide Web Conference 2024 (WWW’24).

The complete code will be updated after the paper is accepted.


# CTR_GNN
This is an innovative program of ctr with gnn.

This code is based on tgin (https://github.com/alibaba/tgin) which is an inheritance from DMIN (https://github.com/mengxiaozhibo/DMIN).

# Usage
```python
python script/train.py [hyperparams]
```
Hyperparams list:

* --train: to train the model
* --test: to test the model
* --shuffle: shuffle the dataset before training each epoch
* --seed [value]: random seed
* --model [value]: model name, eg --model GEIN
* --dataset [value]: name of dataset folder, eg --dataset elec_c2c
* --device [value]: gpu id, -1 for cpu
* --num_prod 6: set the number of neighbors of each behavior to 6
* --mask_ratio [value]: ratio of behavior mask augmentation for contrastive learning
* --drop_ratio [value]: ratio of feature dropout augmentation for contrastive learning
* --sub_ratio [value]: ratio of behavior substitude augmentation for contrastive learning
* --ssl_weight [value]: weight of contrastive learning loss
* --use_projector_head: use projector head for contrastive learning
* --iters [value]: training epochs
* --valid_batch [value]: batch number to begin model evaluation on valid dataset
* --lr [value]: learning rate
* --maxlen [value]: max length of user behavior sequence
* --batch_size [value]: batch size

For example:
```python
python script/train.py --test --model GEIN --dataset elec_c2c --num_prod 6 --mask_ratio 0.1 --drop_ratio 0.0 --sub_ratio 0.2 --seed 3 --iters 1 --valid_batch 2400 --maxlen 20 --lr 0.002 --batch_size 128
```

# Required Packages
* python 3.6.13
* tensorflow==1.15.0

# Data Process

To build the dataset for DMINI2I/DMINC2C/DMINC2C_SSL models, please follow these steps:

* Modify the **data_path** variable in the *script/dataset_process/process_c2c.py* file at line 28 to the folder location of your dataset.
* Modify the **processed_path** variable in the same file at line 29 to the location where you want to save the processed dataset.
* Finally, run the following command in your terminal or command prompt:

```python
script/dataset_process/process_c2c.py
```

To build the dataset for GEIN, follow these additional steps:
* Based on the previous steps, modify the **original_path** and **data_path** variables in the *script/dataset_process/process_factor.py* file at lines 355 and 356. Set **original_path** to the path of your original dataset and **data_path** to the location where you save the processed dataset.
Run the following command in your terminal or command prompt:

```python
python script/dataset_process/process_factor.py
```

