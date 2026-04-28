import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

from sklearn.model_selection import GridSearchCV

import os
import pickle
import random

""" This file contains functions that can be generally useful """


#######################################################################################
# General functions                                                                   #
#######################################################################################

def set_seeds(seed):
    """
    set all necessary random seeds to generate reproducible results

    Args:
        seed: random seed - as int
    """

    random.seed(seed)
    np.random.seed(seed)
    # tf.random.set_seed(seed)


def save_dataframe(df, df_save_name, directory='processed_datasets', file='csv', index=False):
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        pass

    if file == 'excel':
        df.to_excel(os.path.join(directory, df_save_name))
    elif file == 'csv':
        df.to_csv(os.path.join(directory, df_save_name), index=index)
    else:
        print('Dataframe %s not safed! Enter file=csv or file=excel.')


class SaveOutput:
    """Class to duplicate print output to both console and file."""
    def __init__(self, file_path):
        self.file = open(file_path, "w")
        self.terminal = sys.stdout  # Save the original stdout

    def write(self, message):
        self.terminal.write(message)  # Print to console
        self.file.write(message)  # Write to file

    def flush(self):
        self.terminal.flush()
        self.file.flush()


#######################################################################################
# Saving and loading models to and from folder                                        #
#######################################################################################

# safe model to folder
def save_model(model, model_name, directory='models'):
    """
    function to save models to folder - only for sklearn and xgboost models

    Args:
        model: trained model that should be saved
        model_name: name of the model - as str
        directory: name of the folder that will be created - as str
    """

    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        pass

    pickle.dump(model, open(os.path.join(directory, model_name), 'wb'))


# load model from folder
def load_model(model_name, directory='models'):
    """
        function to load models from folder - only for sklearn models

        Args:
            model_name: name of the model - as str
            directory: name of the folder where the model is stored in - as str
        """

    pickled_model = pickle.load(open(os.path.join(directory, model_name), 'rb'))
    return pickled_model


