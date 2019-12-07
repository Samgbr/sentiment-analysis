### Sentiment-analysis COMP 479 Project (Machine Learning)

Group Members
1. Samuel Habte
2. Rejoice Jembamalaidass
3. Pranjali Mishra

+ Dataset used is a cleaned IMDB data from [Kaggle link](https://www.kaggle.com/oumaimahourrane/imdb-movie-reviews-cleaned-data)
+ Total dataset is 24902 and for this project randomize the dataset and took 5000 samples. Script file related to this operation is IMDB.ipynb
+ For Sentiment analysis used the bag of words using CountVectorize, Holdout method and different models to fit train dataset and test on validation set and finally with the best performer model test dataset.
+ Models used are Logistics Regression, Support Vector Machines, Adaline SDG(Stochastic Gradient Decent), Perceptron, Naive Bayes (Bernoulli and Multinominal) and KNN. Script file for this implementation is SentimentIMDB.ipynb
+ Deep Learning with Keras with a GloVe(Global Vectors of Word Representation) - Embedded layer with pre-trained word vectors with sigmoid activation function. Script file for this implementation is SentimentKeras.ipynb
  + Simple Neural Network - densely connected neural network
  + Convolutional Neural Network - 1-Dimension and 1-Pooling layer
  + Recurrent Neural Network - LSTM(Long Short Term Memory)
+ Metrics used Accuracy, Confusion Matrix, F1 Score, Recall and Precision
+ Performance report and slide preparation
+ For Embedded layer downloaded the pre-defined word vectors from the [link](https://nlp.stanford.edu/projects/glove/) 
