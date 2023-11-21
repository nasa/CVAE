import os
import torch
import time
import argparse
import logging
from models import *
from utils import *
import numpy as np

cuda = torch.cuda.is_available()
parser = argparse.ArgumentParser()
logger = logging.getLogger("CVAE-Log")
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


parser.add_argument('-l',
                    type=int,
                    default=128,
                    help='latent space dimension')

parser.add_argument('-b',
                    type=float,
                    default=0.1,
                    help='value of hyper-parameter beta')

parser.add_argument('-n',
                    type=int,
                    default=100,
                    help='number of epochs for training')

parser.add_argument('-s',
                    type=int,
                    default=50,
                    help='number of samples for calculating anomaly score')

parser.add_argument('-m',
                    type=str,
                    default='BCE',
                    choices=['BCE', 'MSE'],
                    help='metric to be used for the reconstruction loss, e.g., BCE or MSE')

parser.add_argument('-bs',
                    type=int,
                    default=128,
                    help='size of the mini-batches')


args = parser.parse_args()


"""
Define constants and input arguments
"""
logger.info("initializing...")
num_epochs = args.n
latent_dim = args.l
beta = args.b
num_sample = args.s
metric = args.m
batch_size = args.bs

current_directory = os.getcwd()
dir_2_data = current_directory + "/data/"
saving_dir = current_directory + "/output/"

"""
Loading the data
"""
# the data should be 
## 1- normalized, if you normalize using minmax scaling, you can
##### use any of the metrics, but if you use standard scaling,
##### only use MSE as a metric, since BCE won't be applicable.

## 2- divided into three sets of training, validation, and testing

## 3- data format should be in #_instances * window_size * num_param

logger.info("loading the data...")
full_data = np.load(dir_2_data+"DASHlink_binary_Flaps_noAnomaly_github.npz")
x_train = full_data['train_data']
x_valid = full_data['valid_data']
x_test = full_data['test_data']

model_name = ("CVAE_l_"+(str(latent_dim))+"_beta_"+(str(beta))+
              "_batch_"+str(batch_size)+"_metric_"+metric)

print(model_name)

# In this part, we identify whether the data is normalized using minmax scaler
## or standard scaler. This will be used for parameterizing the model, specifically
### in the last layer of the decoder, where we use sigmoid activation.
min_value = np.min(x_train)
max_value = np.max(x_train)
if max_value > 1:
    scale_flag = 1
    if metric == 'BCE':
        logger.info("Due to standard scaling in data, only MSE can be used as a metric")
    metric = 'MSE'
elif min_value < 0:
    scale_flag = 1
    if metric == 'BCE':
        logger.info("Due to standard scaling in data, only MSE can be used as a metric")
    metric = 'MSE'
else:
    scale_flag = 0

# first dimension in each data instance should be window size
window_size = np.shape(x_train)[1]
# second dimension in each data instance should be parameters (variables)
num_param = np.shape(x_train)[2]

x_train = np.transpose(x_train, axes=(0, 2, 1))
x_valid = np.transpose(x_valid, axes=(0, 2, 1))
x_test = np.transpose(x_test, axes=(0, 2, 1))

# creating dataset instances for training and validation
train = Dataset(x_train)
validation = Dataset(x_valid)

# creating dataloader for training
train_data, valid_data = get_dataset(train, validation, batch_size)


"""
Building the model
"""
logger.info("building the model...")

model = VAE(latent_dim, num_param, window_size, scale_flag)
if cuda: model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.999))


"""
Training the model
"""
logger.info("training...")
model = train_model(model, optimizer, model_name, train_data, valid_data,
                    saving_dir, metric, beta, num_epochs, save=True, verbose=1)
model = model.to('cpu')


"""
Finding threshold for anomaly detection
"""
logger.info("finding anomaly score threshold...")
train_anomaly_score = find_score(model, x_train, metric, num_sample)

# we set the threshold to the mean + 2*std of training anomaly scores
## this can be revised based on the domain feedback
threshold = np.mean(train_anomaly_score) + 2 * np.std(train_anomaly_score)


"""
Test the model
"""
logger.info("testing the model...")
model_preds = np.zeros(np.shape(x_test)[0])
test_anomaly_score = find_score(model, x_test, metric, num_sample)
model_preds[test_anomaly_score > threshold] = 1
np.savez_compressed((saving_dir+model_name+"test_anomaly_labels"), model_preds=model_preds)