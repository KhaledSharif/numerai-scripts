# Numerai Machine Learning Challenge

<i>Markets become more efficient over time. In the 1970s the market seemed efficient. But to Ed Thorp, the market was inefficient because he had invented something new to price options. Many innovations since then have also increased market efficiency. Today, advances in artificial intelligence are poised to transform the global asset management industry.</i>

This repository contains my continuing work on the <i><a href="http://numer.ai/">Numer.ai</a></i> machine learning challenge. Numerai is a global artificial intelligence tournament to predict the stock market, by providing encrypted stock market data and asking the competitor to perform binary classification. 


The training data is presented in 15 features, scaled up to 120 with simple feature preproccesing, and is as follows: the first fourteen columns (f1 - f14) are integer features; column c1 is a categorical feature; column validation indicates a dataset that you can use to validate your model; column target is the binary class you’re trying to predict. The validation set is a subset of the training data that is supposed to be more representative of the tournament data than a random sample of training data would be.



<img src="http://i.imgur.com/TWjwvGb.png" />
<i>The graph above shows the resulting AUC's after each epoch of training <a href="https://github.com/KhaledSharif/numerai-scripts/blob/master/lasagne-script.py">a deep neural network</a>, consisting of 8 highway layers, and built using the Lasange library for Python. The maximum AUC is ~54.5%, and is achieved after 12 iterations.</i>
