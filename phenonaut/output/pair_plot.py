# Copyright © The University of Edinburgh, 2022.
# Development has been supported by GSK.

from phenonaut.data import Dataset


def show_pair_plot(self, ds: Dataset):
    """Visualise the dataset using a pairplot.

    Parameters
    ----------
    ds : Dataset
        The dataset to visualise
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.pairplot(ds.df)
    plt.show()
