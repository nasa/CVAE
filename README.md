# CVAE

This repository contains the source code (in the source directory) for Convolutional Variational Auto-Encoder, CVAE [1]. CVAE takes advantage of the convolutional neural networks to learn about the short-term as well as long-term temporal dependence in the input data. This is in comparison to fully connected neural networks that won't incorporate such temporal dependece. As a result, it is well-suited to work with multivariate time-series data. CVAE can be used for multiple purposes such as dimensionality reduction, anomaly detection, and data generation. 

In the folder *examples* we provide a jupyer notebook, explaining a detailed implementation of CVAE for a task of anomaly detection in multivariate time-series of flight's operational data (the sample data is in the *data* folder). The source code *CVAE.py* can be edited accordingly to deploy CVAE on other tasks mentioned above for any multi-variate time series data. You can run a fast episode of training CVAE (only 10 epochs) for anomaly detection (similar to the one in the jupyter notebook example) via the following command:

```
python source/CVAE.py -l 32 -n 10 -s 5
```

[1] Memarzadeh, M., Matthews, B., and Avrekh, I. (2020). Unsupervised Anomaly Detection in Flight Data Using Convolutional Variational Auto-Encoder. *Aerospace*, 7(8), 115, https://doi.org/10.3390/aerospace7080115.
