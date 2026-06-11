import pandas as pd
import numpy as np

train_x = pd.DataFrame(np.load('../Datasets/raw/AppClassNet/top200/train_x.npy')).to_csv('../Datasets/converted/train_x.csv')
train_y = pd.DataFrame(np.load('../Datasets/raw/AppClassNet/top200/train_y.npy')).to_csv('../Datasets/converted/train_y.csv')

test_x = pd.DataFrame(np.load('../Datasets/raw/AppClassNet/top200/test_x.npy')).to_csv('../Datasets/converted/test_x.csv')
test_y = pd.DataFrame(np.load('../Datasets/raw/AppClassNet/top200/test_y.npy')).to_csv('../Datasets/converted/test_y.csv')

valid_x = pd.DataFrame(np.load('../Datasets/raw/AppClassNet/top200/valid_x.npy')).to_csv('../Datasets/converted/valid_x.csv')
valid_y = pd.DataFrame(np.load('../Datasets/raw/AppClassNet/top200/valid_y.npy')).to_csv('../Datasets/converted/valid_y.csv')