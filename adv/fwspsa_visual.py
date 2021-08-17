# %%
import os
import pickle
import matplotlib.pyplot as plt

# %%

batch_1 = pickle.load(open(r'sucess_rate_22_15_15.pkl', 'rb'))
plt.plot(batch_1, label='batch 1')
plt.legend()
plt.show()
