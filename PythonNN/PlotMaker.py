import matplotlib.pyplot as plt


def make_plot(epochs, TR_data, TS_data, ylabel, title, filepath, ylim=None):

    plt.plot(epochs, TR_data, '--', color="b", label="Training")
    plt.plot(epochs, TS_data, color="r", label="Test")
    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt.title(title)
    if (ylim != None):
        plt.ylim(ylim[0], ylim[1])

    plt.savefig(filepath)
    # plt.show()
    plt.close()
