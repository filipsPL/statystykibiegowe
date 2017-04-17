import numpy as np
import seaborn as sns
sns.set(style="darkgrid", palette="Set2")

gammas = sns.load_dataset("gammas")
print gammas
exit(1)

# Create a noisy periodic dataset
sines = []
rs = np.random.RandomState(8)
for _ in range(2):
    x = np.linspace(0, 30 / 2, 30)
    y = np.sin(x) + rs.normal(0, 1.5) + rs.normal(0, .3, 30)
    sines.append(y)

# Plot the average over replicates with bootstrap resamples
print sines
sns.tsplot(sines, err_style="boot_traces", n_boot=500)
#sns_plot.savefig("ts.png")
sns.plt.show()