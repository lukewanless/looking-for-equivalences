import matplotlib.pyplot as plt

def plot_error_bar(x, y, yerr, color1, color2, label):
    plt.plot(x, y, color=color1, linewidth=1)
    plt.errorbar(x=x, y=y, yerr=yerr,
                 color=color1, fmt='o',
                 ecolor=color2, elinewidth=1, label=label)