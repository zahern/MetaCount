import numpy as np
import pandas as pd
import random
import scipy.stats as ss
from scipy.stats import maxwell
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri, numpy2ri
import anndata2ri

anndata2ri.activate()
r = robjects.r

numpy2ri.activate()
#anndata2ri.activate()
r['source']('CONWAY_MODEL_R.R')
conway_function_r = robjects.globalenv['CONWAY']


def noise(n_obs, perc=.2):
    noise_vec = np.zeros(n_obs)
    rand_pos = np.random.randint(n_obs, size=(int(n_obs * perc)))
    noise_vec[rand_pos] = np.random.normal(size=(int(n_obs * perc)))
    return noise_vec


np.random.seed(0)
N = 4000  # Number of observations

# Generate input data
df = pd.DataFrame()
df['id'] = np.arange(1, N + 1)

df['postcode'] = random.choices(range(4000, 4400), k=N)
df['const'] = np.ones(N)
df['aadt'] = random.choices(range(300, 3000), k=N)
df['aadt'] = np.log(df['aadt'])
df['curve'] = random.choices([0.1, 0.2, 0.3, 0.4], k=N) + noise(N)
df['speed'] = random.choices([50, 60, 70, 80, 90, 100, 110], weights=[1, 2, 1, 3, 2, 5, 2], k=N)
df['speed'] = np.log(df['speed'])
df['lanes'] = random.choices(range(1, 6), k=N)
df['medwidth'] = random.choices(range(1, 7), k=N) + noise(N)
df['day'] = random.choices([0, 1], k=N)
df['carriageway'] = random.choices([1, 2, 3, 4], k=N)
df['vehicle_car'] = random.choices(range(0, 2), k=N)

df['non1'] = random.choices(range(1, 100), k=N) + noise(N)
df['non2'] = random.choices(range(1, 50), k=N) + noise(N)
df['non3'] = random.choices(range(20, 30), k=N) + noise(N)
df['non4'] = random.choices(range(4, 9), k=N) + noise(N)
df['non5'] = random.choices(range(0, 2), k=N)
df['non6'] = random.choices(range(0, 2), k=N)
df['non7'] = random.choices(range(0, 2), k=N)
df['non8'] = random.choices(range(0, 2), k=N)
df = df.round(3)

# Define coefficients (betas)
# Fixed betas
Bconst, Baadt, Bcurve, Bspeed, Blanes, Bmedwidth = -4, 1.1, 0.15, 1.3, 0.2, -4

# Random betas
Bdayr = np.random.normal(loc=-4.5, scale=1, size=N)
Bcarggr = np.random.normal(loc=-5, scale=1, size=N)
Bvehicler = np.random.normal(loc=-2.5, scale=1, size=N)

# Convert betas to matrix for easy product
B = [np.repeat(Bconst, N), np.repeat(Baadt, N), np.repeat(Bcurve, N), np.repeat(Bspeed, N), np.repeat(Blanes, N),
     np.repeat(Bmedwidth, N), np.repeat(Bdayr, 1), np.repeat(Bcarggr, 1),
     np.repeat(Bvehicler, 1), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N),
     np.zeros(N)]
B = np.vstack(B).T

# Multiply and generate probability
X = df.values[:, 2:]  # Extract only necessary columns

XB = (X * B).sum(axis=1).reshape(N, 1)
eXB = np.exp(XB)
eXB = np.array(eXB).transpose()


XB = np.array(XB).transpose()

try:

    mean_6 = conway_function_r(eXB, 0.5)
    print(mean_6.sum())
    #df_result = pandas2ri.py2ri(mean_4)
    print(1)
    df['y'] = mean_6
except Exception as e:
    print(e)
    print('wg')



# Save to CSV
df.to_csv("artificial.csv", index=False)