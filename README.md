SharpestMinds-Application: ground-cover-classification project

This is a project to classify the type of ground cover in a surveyed area from several other pieces of data. The data that we have consists of things like elevation, distance to water, etc. and the label to be determined is one of seven types of ground cover like spruce, or aspen trees.

My starting point was the paper 'Comparative accuracies of artificial neural networks and discriminant analysis in predicting forest cover types from cartographic variables' by Jock A.Blackard, and Denis J.Dean. They implement a network with one hidden layer and optimize for the number of nodes in that layer and the learning rate.
First I replicated their result, which achieves about 70% accuracy. I experimented with sampling from the data using different probability distributions in the notebookes 'classifier-natural' and 'classifier-uniform'. in My goal was then to improve upon this result with deeper networks and/or other machine learning techniques.

I explored decision trees and random forests in the notebook 'decision_tree_classifier-experiments', and with a random forest achieved an accuracy of about 94.5%. In this notebook I also experimented with gradient boosting classifiers, but the computation time was very long. I also tried some data augmentation, but it offered no increased performance. 
I consolidated the code to generate the 1000 tree random forest into the notebook 'decision_tree_classifier'. (Saving this random forest requires a very large amount of memory, so it was added to .gitignore.)

I was pretty impressed with the performance of the random forest, but set out to try to beat it with a deep neural net in the notebook 'classifier-natural-deep-experiments'. Using model averaging of 10 neural nets with several hidden layers and batch normalization, I achieved an accuracy of about 96.5%, a 2% increase over the random forest. I consolidated the code to generate those nets in 'classifier-natural-deep-model-averaging'. I was curious if there would be noticable gains in accuracy from make each data field have mean zero, but didn't see any improvement ('classifier-natural-deep-experiments-mean-zero'). I also tried including some dropout, but it didn't improve performance. To keep the repository size down, I added the saved models to .gitignore, but if you would like to have access to them, I can arrange it.

Finally, in 'comparison-of-random-forest-and-model-averaged-nn', I started to take a look at which examples my two main techniques were getting wrong. My goal was to come up with a way to combine them to produce an ever more accurate hybrid model, but haven't made much progess on it at this point. There is a tiny bit of data exploration in the 'data-exploration' notebook, but nothing os particular note.

The data came in a nice format, so the data_preprocessing and utils libraries are largely straightforward. 


Last month I applied with a project in a repository now called Bach-Chorales-Generative where I used a LSTM network to predict the next note given an initial phrase to write chorales in the style of Bach. That project had a fair bit more preprocessing than my new project due to the formatting of the raw data files, but the project didn't lend itself to seeing the full process of solving a problem with machine learning. Hopefully this new project does a better job of displaying this process. The feedback that I received for that project follows:

"Really solid application, it's clear that he knows how to use preprocessing tools and the Keras API. However, it's always hard to gauge actual machine learning capabilities through a project this challenging when it doesn't succeed - would recommend taking on a problem that can be successfully solved (even if it's a bit simpler) so that we can see the end-to-end thought process a bit more clearly. Also would like to see PEP-8 adhered to a little more closely (particularly with the inclusion of brief docstrings for every non-intuitive function). With a fully worked-out problem to review, would probably be a shoe-in for next batch."

