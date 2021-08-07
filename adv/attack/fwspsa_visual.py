# %%
import os
import pickle
import matplotlib.pyplot as plt

# %%

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
batch_1 = pickle.load(open(r'D:\__YBR\Study\Courses\AI2612_MachineLearningProject\Repository\adv\19_55_45.pkl', 'rb'))
batch_2 = pickle.load(open(r'D:\__YBR\Study\Courses\AI2612_MachineLearningProject\Repository\adv\19_56_17.pkl', 'rb'))
batch_3 = pickle.load(open(r'D:\__YBR\Study\Courses\AI2612_MachineLearningProject\Repository\adv\19_56_49.pkl', 'rb'))
plt.plot(batch_1, label='batch 1')
plt.plot(batch_2, label='batch 2')
plt.plot(batch_3, label='batch 3')
print(len(batch_1))
plt.legend()
plt.show()
