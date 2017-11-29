SharpestMinds-Application: ground-cover-classification project

This is a project to classify the type of ground cover in a surveyed area from several other pieces of data. The data that we have consists of things like elevation, distance to water, etc. and the label to be determined is one of seven types of ground cover like spruce, or aspen trees.

My starting point was the paper 'Comparative accuracies of artificial neural networks and discriminant analysis in predicting forest cover types from cartographic variables' by Jock A.Blackard, and Denis J.Dean. They implement a network with one hidden layer and optimize for the number of nodes in that layer and the learning rate.
First I replicated their result, which achieves about 70% accuracy. My goal was then to improve upon this result with deeper networks and/or other machine learning techniques.

I explored decision trees, and with a random forest achieved an accuracy of about 96%.

Using model averaging of neural nets with several hidden layers, I achieved an accuracy just shy of 97%.


Last month I applied with a project in a repository now called Bach-Chorales-Generative where I used a LSTM network to predict the next note given an initial phrase to write chorales in the style of Bach. That project had a fair bit more preprocessing than my new project due to the formatting of the raw data files, but the project didn't lend itself to seeing the full process of solving a problem with machine learning. Hopefully this new project does a better job of displaying this process. The feedback that I received for that project follows:

"Really solid application, it's clear that he knows how to use preprocessing tools and the Keras API. However, it's always hard to gauge actual machine learning capabilities through a project this challenging when it doesn't succeed - would recommend taking on a problem that can be successfully solved (even if it's a bit simpler) so that we can see the end-to-end thought process a bit more clearly. Also would like to see PEP-8 adhered to a little more closely (particularly with the inclusion of brief docstrings for every non-intuitive function). With a fully worked-out problem to review, would probably be a shoe-in for next batch."

