import numpy as np
from keras.utils import to_categorical

def shuffle_in_unison(a, b):
    """Shuffle the entries of two arrays using the same permutation."""
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
    

def percent_correct(model,data,labels):
    """Returns the percentage of correct predicted labels."""
    predict = model.predict(data)
    num_cor = 0
    for i in np.arange(len(labels)):
        if np.argmax(predict[i]) == np.argmax(labels[i]):
            num_cor = num_cor + 1
    print('Correct:',num_cor*100/len(labels),'%')
    
def get_batch(data, labels, num = 10000):
    """Gets a batch consisting of num (default 10000) samples randomly chosen with replacement
    from the input data and labels.
    """
    
    indices = np.random.choice(data.shape[0]-1,num)
    batch_data = data[indices,:]
    batch_labels = labels[indices,:]
    
    return batch_data, batch_labels
    
def get_uniform_batch(data, labels, num_per_label = 1000):
    """If you are training on data that is not uniformly distributed among the labels, this function
    makes a batch with equal numbers of each label sampled with replacement from
    the input data and labels. The batch will have size 7*num_per_label elements, default 7000.
    """
    full_batch = []
    
    data_labs = np.concatenate((data,labels),axis=1)
    for i in np.arange(0,7):
        data_set = data_labs[np.where((labels==to_categorical(i,7)).all(axis=1))[0],:]
        indices = np.random.choice(data_set.shape[0]-1,num_per_label)
        full_batch.extend(data_set[indices,:])
    full_batch = np.array(full_batch).reshape(7*num_per_label,data_set.shape[1])
    
    batch_data = full_batch[:,:-7]
    batch_labels = full_batch[:,-7:]
        
    return batch_data, batch_labels

def get_unbalanced_batch(data, labels, num_per_label = [100,100,100,100,100,100,6400]):
    """A function to create a batch with a given number of samples with each label."""
    full_batch = []
    
    data_labs = np.concatenate((data,labels),axis=1)
    for i in np.arange(0,7):
        data_set = data_labs[np.where((labels==to_categorical(i,7)).all(axis=1))[0],:]
        indices = np.random.choice(data_set.shape[0]-1,num_per_label[i])
        full_batch.extend(data_set[indices,:])
    full_batch = np.array(full_batch).reshape(np.sum(num_per_label),data_set.shape[1])
    
    batch_data = full_batch[:,:-7]
    batch_labels = full_batch[:,-7:]
    
    return batch_data, batch_labels
