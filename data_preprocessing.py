import pandas as pd
from keras.utils import to_categorical
from utils import shuffle_in_unison

def import_data(mean_zero = False):
    """Imports the data from file and returns the a dataframe, df, containing the data,
    scaled so that each entry is between 0 and 1."""
    
    # Import from file
    df = pd.read_table('data/covtype.data' , sep = ',' , header = None)
    
    # Extract the last column, which corresponds to the labels
    labels = df.iloc[:,-1]
    # Remove the last column, which corresponds to the labels
    df = df.drop(df.columns[-1],axis=1)

    # Normalize each column to be between 0 and 1 
    df = df - df.min()
    df = df / df.max()
    
    if mean_zero == True:
        # Make each column have mean zero
        df = df - df.mean()
    
    # Put the label column back on
    df = pd.concat([df, labels], axis=1)
    
    return df


def get_data_sets(mean_zero = False, seed = None):
    
    """Returns test, validation and training sets and labels.
    Recommended usage:  
    test_data, test_labels, valid_data, valid_labels, train_data, train_labels = get_data_sets()
    """
    
    # Import the data as a dataframe
    df = import_data(mean_zero = False)

    # take about 80% of the data for the training and validation sets
    train_df_size_per_index = 370000 # about 64% of the data
    valid_df_size_per_index = 100000 # about 16% of the data
    
    #Shuffle the dataframe df
    df = df.sample(frac=1, random_state = seed)
    
    # Put the first test_df_size into the test set
    train_df = df[:train_df_size_per_index]
    # Put the next valid_df_size into the validation set
    valid_df = df[train_df_size_per_index:train_df_size_per_index+valid_df_size_per_index]
    # Put the remainder into the training set
    test_df = df[train_df_size_per_index+valid_df_size_per_index:]
    
    # Extract the last columns, which corresponds to the labels
    test_labels = test_df.iloc[:,-1]
    valid_labels = valid_df.iloc[:,-1]
    train_labels = train_df.iloc[:,-1]
    
    # Remove the last columns, which corresponds to the labels
    test_df = test_df.drop(test_df.columns[-1],axis=1)
    valid_df = valid_df.drop(valid_df.columns[-1],axis=1)
    train_df = train_df.drop(train_df.columns[-1],axis=1)
    
    # Convert data from dataframes to np.arrays
    test_data = test_df.values
    valid_data = valid_df.values
    train_data = train_df.values
    test_labels = test_labels.values
    valid_labels = valid_labels.values
    train_labels = train_labels.values
    
    # Convert labels to one hot vectors
    test_labels = to_categorical(test_labels-1,7)
    valid_labels = to_categorical(valid_labels-1,7)
    train_labels = to_categorical(train_labels-1,7)
    
#     # Shuffle the data and labels
#     shuffle_in_unison(test_data, test_labels)
#     shuffle_in_unison(valid_data, valid_labels)
#     shuffle_in_unison(train_data, train_labels)
    
    return test_data, test_labels, valid_data, valid_labels, train_data, train_labels

# def augment_data(data, labels):
#     """This function returns an augmented data set where noise has been added to to the first 10 fields
#     to create a larger collection of data.
#     """
    