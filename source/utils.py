import torch
import math
import time
import numpy as np
from torch.autograd import Variable
cuda = torch.cuda.is_available()


class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        X = self.data[index, :, :]

        return X


def get_dataset(train, validation, batch_size=128):
    """this function creates the dataloader for the training and validation
    data to be used for training.

    Args:
        train (torch dataset): trianing dataset
        validation (torch dataset): validation dataset
        batch_size (int, optional): batch size for training. Defaults to 128.

    Returns:
        dataloaders: training and validation dataloaders
    """
    
    from torch.utils.data.sampler import SubsetRandomSampler

    def get_sampler(data):
        num_data = np.shape(data)[0]
        sampler = SubsetRandomSampler(torch.from_numpy(np.arange(0, num_data)))

        return sampler

    train_data = torch.utils.data.DataLoader(train, batch_size=batch_size, pin_memory=cuda,
                                             sampler=get_sampler(train.data))
    valid_data = torch.utils.data.DataLoader(validation, batch_size=batch_size, pin_memory=cuda,
                                             sampler=get_sampler(validation.data))
    
    return train_data, valid_data


def likelihood_loss(r, x, metric='BCE'):
    """calculates likelihood loss between input and its reconstruction

    Args:
        r (tensor): reconstructed data
        x (tensor): input data

    Returns:
        tensor: likelihood loss between reconstructed and input data
    """
    r = r.view(r.size()[0], -1)
    x = x.view(x.size()[0], -1)

    if metric == 'BCE':
        likelihood_loss = -torch.sum(x * torch.log(r + 1e-8) + (1 - x) * torch.log(1 - r + 1e-8), dim=-1)
    elif metric == 'MSE':
        mse_loss = torch.nn.MSELoss(reduction='none')
        likelihood_loss = torch.sum(mse_loss(x, r), dim=-1)

    return likelihood_loss


def train_model(model, optimizer, model_name, train_data, valid_data, save_dir,
                metric='BCE', beta=1, num_epochs=100, save=True, verbose=1):
    """Trains the model

    Args:
        model (torch neural network model): a model built with torch.nn
        optimizer (torch optimizer): optimizer to be used for training
        model_name (string): name for saving the resulting model
        train_data (torch dataloader): training dataloader
        valid_data (torch dataloader): validation dataloader
        beta (int, optional): parameter beta. Defaults to 1.
        num_epochs (int, optional): number of epochs of training. Defaults to 100.
        save (bool, optional): whether to save the trained model or not. Defaults to True.
        verbose (int, optional): whether to printout the losses during training. Defaults to 1.

    Returns:
        torch model: trained model
    """
    training_total_loss = np.zeros(num_epochs)        # this records total average loss
    training_rec_loss = np.zeros(num_epochs)          # this records the likelihood loss
    training_kl_loss = np.zeros(num_epochs)           # this records the KL loss
    
    for epoch in range(num_epochs):
        start=time.time()
        model.train()
        total_loss, rec_loss, kl_loss = (0, 0, 0)
        for x in train_data:
            
            x = Variable(x)
            if cuda: x = x.cuda(device=0)

            # obtaining the reconstruction for the mini-batch
            reconstruction = model(x)

            # calculating the likelihood with the specified metric by user
            likelihood = -likelihood_loss(reconstruction, x, metric)
    
            # calculate ELBO 
            elbo = likelihood - beta * model.kl_div
            L = -torch.mean(elbo)

            # performing backpropagation and optimizer steps
            L.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            rec_loss += torch.mean(-likelihood).item()
            kl_loss += torch.mean(model.kl_div).item()
            total_loss += L.item()
        
        stop = time.time()
        duration = stop - start
        
        mt = len(train_data)
        training_total_loss[epoch] = total_loss / mt
        training_rec_loss[epoch] = rec_loss / mt
        training_kl_loss[epoch] = kl_loss / mt
        if epoch % 1 == 0:
            if verbose:
                print("Epoch: {},  Duration: {:0.2f}".format(epoch, duration))
                print("[Loss Train]\t\t L: {:.4f}, Rec: {:.4f}, "
                    ", KL: {:.4f}".format(total_loss/mt, rec_loss/mt, kl_loss/mt))
            valid_loss = 0
            for x in valid_data:

                x = Variable(x)
                if cuda: x = x.cuda(device=0)
                reconstruction = model(x)
                likelihood = -likelihood_loss(reconstruction, x, metric)
                elbo = likelihood - beta * model.kl_div
                L_valid = -torch.mean(elbo)
                valid_loss += L.item()
            mv = len(valid_data) 
            if verbose:
                print("[Loss Validation]\t\t L: {:.4f}".format(valid_loss/mv))
    
    if save:
        torch.save(model.state_dict(), (save_dir+model_name+".pth"))
        np.savez_compressed((save_dir+model_name+"_training_loss"), training_total_loss=training_total_loss,
                            training_rec_loss=training_rec_loss,
                            training_kl_loss=training_kl_loss)

    return model


def find_score(model, data, metric='BCE', num_sample=50):
    """finds the anomaly score

    Args:
        model (torch model): trained model
        data (matrix): input data
        num_sample (int, optional): number of times to reconstruct data. Defaults to 50.

    Returns:
        vector: reconstruction error for each data point. 
    """
    model.eval()
    num_data = np.shape(data)[0]
    data_tensor = torch.tensor(data).float()
    anomaly_score = np.zeros((num_data, num_sample))
    for i in range(num_sample):
        reconstruction = model(data_tensor)
        lh_loss = likelihood_loss(reconstruction.reshape((reconstruction.size(0), -1)),
                                              data_tensor.reshape((data_tensor.size(0), -1)),
                                              metric)
        anomaly_score[:, i] = lh_loss.detach().numpy()
    avg_anomaly_score = np.mean(anomaly_score, axis=1)
    return avg_anomaly_score
