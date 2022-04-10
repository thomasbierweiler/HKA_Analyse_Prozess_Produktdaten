# tested with python 3.8.10 and ydata_synthetic version 0.3.0
# for ydata_synthetic: use python < 3.9

# Author: Archit Yadav
# Source: https://towardsdatascience.com/modeling-and-generating-time-series-data-using-timegan-29c00804f54d
# Modeling and Generating Time-Series Data using TimeGAN
# Case study: Energy Dataset

# data set see https://github.com/jsyoon0823/TimeGAN/blob/master/data/energy_data.csv

# We take a window of size 24 and run it along the rows of the dataset,
# shifting by one position at a time and hence obtaining a certain number of 2D matrices,
# each having a length of 24 and with all our coloumn features.

# We first read the energy dataset and then apply some pre-processing in the form of data transformation.
# This pre-processing essentially scales the data in the range [0, 1]

seq_len=24 # number of timesteps

import pandas as pd
import numpy as np
from ydata_synthetic.preprocessing.timeseries.utils import real_data_loading

file_path = "./data/energy_data.csv"
energy_df = pd.read_csv(file_path)

try:
    energy_df = energy_df.set_index('Date').sort_index()
except:
    energy_df=energy_df

# Data transformations to be applied prior to be used with the synthesizer model
energy_data = real_data_loading(energy_df.values, seq_len=seq_len)

print(len(energy_data), energy_data[0].shape)

# Now generating the actual synthetic data from this time-series data
# (energy_data) is the simplest part.
# We essentially train the TimeGAN model on our energy_data and
# then use that trained model to generate more.

from ydata_synthetic.synthesizers.timeseries import TimeGAN

# Where the parameters to be fed to TimeGAN constructor have to be defined
# appropriately according to our requirements. We have n_seq defined as 28 (features),
# seq_len defined as 24 (timesteps). The rest of the parameters are defined as follows:

seq_len = 24        # Timesteps
n_seq = 28          # Features

hidden_dim = 24     # Hidden units for generator (Gated Recurrent Unit (GRU) & LSTM).
                    # Also decides output_units for generator

gamma = 1           # Used for discriminator loss

noise_dim = 32      # Used by generator as a starter dimension
dim = 128           # UNUSED
batch_size = 128

learning_rate = 5e-4
beta_1 = 0          # UNUSED
beta_2 = 1          # UNUSED
data_dim = 28       # UNUSED

# batch_size, lr, beta_1, beta_2, noise_dim, data_dim, layers_dim
gan_args = (batch_size, learning_rate, beta_1, beta_2, noise_dim, data_dim, dim)

synth = TimeGAN(model_parameters=gan_args, hidden_dim=hidden_dim, seq_len=seq_len, n_seq=n_seq, gamma=1)
synth.train(energy_data, train_steps=500)
synth.save('synth_energy.pkl')

synth_data = synth.sample(len(energy_data))

# Evaluation and Visualization
# We can make use of the following two well know visualization techniques:
# PCA — Principal Component Analysis
# t-SNE — t-Distributed Stochastic Neighbor Embedding

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sample_size = 250
idx = np.random.permutation(len(energy_data))[:sample_size]

# Convert list to array, but taking only 250 random samples
# energy_data: (list(19711(ndarray(24, 28)))) -> real_sample: ndarray(250, 24, 28)
real_sample = np.asarray(energy_data)[idx]
synthetic_sample = np.asarray(synth_data)[idx]

# For the purpose of comparison we need the data to be 2-Dimensional.
# For that reason we are going to use only two components for both the PCA and TSNE.
# synth_data_reduced: {ndarray: (7000, 24)}
# energy_data_reduced: {ndarray: (7000, 24)}
synth_data_reduced = real_sample.reshape(-1, seq_len)
energy_data_reduced = np.asarray(synthetic_sample).reshape(-1,seq_len)

n_components = 2
pca = PCA(n_components=n_components)
tsne = TSNE(n_components=n_components, n_iter=300)

# The fit of the methods must be done only using the real sequential data
pca.fit(energy_data_reduced)

# pca_real: {DataFrame: (7000, 2)}
# pca_synth: {DataFrame: (7000, 2)}
pca_real = pd.DataFrame(pca.transform(energy_data_reduced))
pca_synth = pd.DataFrame(pca.transform(synth_data_reduced))

# data_reduced: {ndarray: (14000, 24)}
data_reduced = np.concatenate((energy_data_reduced, synth_data_reduced), axis=0)

# tsne_results: {DataFrame: (14000, 2)}
tsne_results = tsne.fit_transform(data_reduced)
df_subset=pd.DataFrame()
df_subset['tsne-2d-one'] = tsne_results[:,0]
df_subset['tsne-2d-two'] = tsne_results[:,1]
# set label
df_subset['y']=''
df_subset.loc[:len(energy_data_reduced),'y']='Original'
df_subset.loc[len(energy_data_reduced)+1:,'y']='Synthetic'
# plot tsne_results
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("hls", 2),
    data=df_subset,
    legend="full",
    alpha=0.3
)
plt.show()
