from sklearn import tree
from data_preprocessing import get_data_sets
import time
import graphviz 
print('Imports Complete')

# Import data
test_data, test_labels, valid_data, valid_labels, train_data, train_labels = get_data_sets()
print('Data Imported')

# Create and train tree
clf = tree.DecisionTreeClassifier()
t = time.time()
clf = clf.fit(train_data, train_labels)
elapsed = time.time() - t
print('Tree constructed in {:0.2f} seconds.'.format(elapsed))

print('Mean accuracy on training set = {:0.2f}%'.format(clf.score(train_data,train_labels)*100))
print('Mean accuracy on validation set = {:0.2f}%'.format(clf.score(valid_data,valid_labels)*100))
print('Mean accuracy on testing set = {:0.2f}%'.format(clf.score(test_data,test_labels)*100))

dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("ground_cover_graphviz") 
print('Complete')