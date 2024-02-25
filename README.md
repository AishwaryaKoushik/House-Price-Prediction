# House-Price-Prediction

This project primarily focusses on building a neural network to predict the price of a house. The dataset spans over a wide range of house features like number of bedrooms and bathrooms, arae of the house, location of the house, distance of the nearby airport etc. We are all aware of the libraries like PyTorch, TensorFlow etc that have built-in neural networks. But here, a neural network is built from scratch without using any of these libraries.

**Salient Features**

The multiple features in the dataset demand pre-processing to be performed to filter out the training data. As the first step, Principal Component Analysis(PCA) is used. PCA helps in identifying the key features and points out those which can be excluded. The dataset with the irrelevant feaures dropped serves as the input to the Neural network.


![PCA](https://github.com/AishwaryaKoushik/House-Price-Prediction/assets/161193220/96dfe539-40ae-43b7-8945-06ab6eea0936)


The type of the neural network has a direct impact on the model's performance. To compare and evaluate, neural networks with no hidden layers, 1 hidden layer and 2 hidden layers are built and tested for accuracy. It is inferred that the neural network with no hidden layers has the least amount of test loss.
