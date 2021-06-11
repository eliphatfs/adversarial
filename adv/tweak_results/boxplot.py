import matplotlib.pyplot as plotlib
plotlib.style.use('seaborn')
plotlib.rcParams['ps.useafm'] = True
fsize = 18
tsize = 22
parameters = {'axes.labelsize': tsize, 'axes.titlesize': tsize,
              'xtick.labelsize': fsize, 'ytick.labelsize': fsize,
              'legend.fontsize': fsize}
plotlib.rcParams.update(parameters)
# plotlib.subplot(3, 1, 1)
plotlib.boxplot([eval(open('o%d.csv' % i).read())[0::3] for i in [1, 2, 3, 4, 5, 6]], showfliers=False)
'''plotlib.subplot(3, 1, 2)
plotlib.boxplot([eval(open('b%d.csv' % i).read())[1::3] for i in [1, 3, 6]], showfliers=False)
plotlib.subplot(3, 1, 3)
plotlib.boxplot([eval(open('b%d.csv' % i).read())[2::3] for i in [1, 3, 6]], showfliers=False)'''
plotlib.legend()
plotlib.xlabel("Model ID", labelpad=16)
plotlib.ylabel("Max. Activation", labelpad=16)
plotlib.savefig("box_plot_spectra.pdf", bbox_inches="tight")
plotlib.show()
