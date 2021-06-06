import numpy
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plotlib


plotlib.style.use('seaborn')
plotlib.rcParams['ps.useafm'] = True
fsize = 18
tsize = 22
parameters = {'axes.labelsize': tsize, 'axes.titlesize': tsize,
              'xtick.labelsize': fsize, 'ytick.labelsize': fsize,
              'legend.fontsize': fsize}
plotlib.rcParams.update(parameters)


x = numpy.linspace(-5, 5, 201)
plotlib.xlabel("$\Delta z$", labelpad=16)
plotlib.ylabel("Activation", labelpad=16)


def plot_w(w):
    fx = torch.tensor(x * w)
    fy = F.sigmoid(fx)
    plotlib.plot(x, fy, label='w = %.1f' % w)


for w in [0.3, 1, 3, 9]:
    plot_w(w)
plotlib.legend()
plotlib.savefig("bin_amp_geo.pdf", bbox_inches="tight")
plotlib.show()
