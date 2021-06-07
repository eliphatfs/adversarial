# %%
import pickle
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.style.use('seaborn')

ce_pure, ce_lost, adamp_pure, adamp_lost = pickle.load(
    open('./track_lost.pkl', 'rb'))

# %%
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

axs[0].set_title('CE Loss Distribution')
sns.distplot(np.log(ce_pure), label='attack failed', ax=axs[0])
sns.distplot(np.log(ce_lost), label='attack succeeded', ax=axs[0])
axs[0].legend()
axs[0].set(xlabel='Value (Log)',)

axs[1].set_title('AdAmp Distribution')
sns.distplot(np.log(adamp_pure), label='attack failed', ax=axs[1])
sns.distplot(np.log(adamp_lost), label='attack succeeded', ax=axs[1])
axs[1].legend()
axs[1].set(xlabel='Value (Log)',)

fig.savefig('CE_AdAmp_distribution.png', dpi=300)
fig.savefig('CE_AdAmp_distribution.pdf')
