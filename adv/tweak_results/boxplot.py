import matplotlib.pyplot as plotlib
plotlib.style.use('seaborn')
plotlib.subplot(3, 1, 1)
plotlib.boxplot([eval(open('b%d.csv' % i).read())[0::3] for i in [1, 3, 6]], showfliers=False)
plotlib.subplot(3, 1, 2)
plotlib.boxplot([eval(open('b%d.csv' % i).read())[1::3] for i in [1, 3, 6]], showfliers=False)
plotlib.subplot(3, 1, 3)
plotlib.boxplot([eval(open('b%d.csv' % i).read())[2::3] for i in [1, 3, 6]], showfliers=False)
plotlib.show()