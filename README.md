# Quant Concepts

In this repository, I keep notes related to my study of topics related to quantitative finance. Most notes are in the form of Jupyter notebooks, and each note is intended to be a self-contained exploration of a topic.

## Table of Contents

- [Regression](#regression)
- [Timeseries](#timeseries)
- [Stochastic Processes](#stochastic-processes)
- [Convolutional & Recurrent Neural Networks](#convolutional--recurrent-neural-networks)
- [Deep Reinforcement Learning](#deep-reinforcement-learning)
- [Machine Learning](#machine-learning)

## Regression

1. [Ordinary Least Squares (OLS)](1.%20Regression/1.%20OLS.ipynb)
2. [Bayesian Linear Regression](1.%20Regression/2.%20Bayesian%20Linear%20Regression.ipynb)
3. [Optimization](1.%20Regression/3.%20Optimization.ipynb)
4. [Ridge & Lasso Regression](1.%20Regression/4.%20Ridge%20&%20Lasso%20Regression.ipynb)
5. [Logistic Regression & Multinomial Logistic Regression](1.%20Regression/5.%20Logistic%20Regression%20&%20Multinomial%20Logistic%20Regression.ipynb)
6. [Generalized Linear Models (GLMs)](1.%20Regression/6.%20Generalized%20Linear%20Models%20(GLMs).ipynb)

## Timeseries

1. [Stationarity](2.%20Timeseries/1.%20Stationarity.ipynb)
2. [ARIMA & SARIMA](2.%20Timeseries/2.%20ARIMA%20&%20SARIMA.ipynb)
3. [ARCH & GARCH](2.%20Timeseries/3.%20ARCH%20%26%20GARCH.ipynb)
4. [MCMC for Parameter Estimation in a Regime-Switching Gaussian Mixture Model](2.%20Timeseries/4.%20MCMC%20for%20Parameter%20Estimation%20in%20a%20Regime-Switching%20Gaussian%20Mixture%20Model.ipynb)
<!-- 4. [Kalman Filters](2.%20Timeseries/4.%20Kalman%20Filters.ipynb) -->

## Stochastic Processes

1. [Brownian Motion](3.%20Stochastic%20Processes/1.%20Brownian%20Motion.ipynb)
2. [Brownian Motion with Drift](3.%20Stochastic%20Processes/2.%20Brownian%20Motion%20with%20Drift.ipynb)

## Deep Neural Networks

1. [MLP Classifier from Scratch in JAX](4.%20Deep%20Neural%20Networks/1.%20MLP%20Classifier%20from%20Scratch%20in%20JAX.ipynb)
2. [MLP Classifier in Numpy](4.%20Deep%20Neural%20Networks/1.1%20MLP%20Classifier%20in%20Numpy.ipynb)
3. [Number Recognition with Keras](4.%20Deep%20Neural%20Networks/2.%20Number%20Recognition%20with%20Keras.ipynb)
4. [Regularization Techniques](4.%20Deep%20Neural%20Networks/3.%20Regularization%20Techniques.ipynb)

## Convolutional & Recurrent Neural Networks

1. [Image Classification with a Convolutional Neural Network in PyTorch](5.%20Convolutional%20&%20Recurrent%20Neural%20Networks/1.%20Image%20Classification%20with%20a%20Convolutional%20Neural%20Network%20in%20PyTorch.ipynb)
2. [RNNs, LSTMs, GRUs](5.%20Convolutional%20&%20Recurrent%20Neural%20Networks/2.%20RNNs,%20LSTMs,%20GRUs.ipynb)

## Deep Reinforcement Learning

1. [Introduction to Deep Reinforcement Learning](6.%20Deep%20Reinforcement%20Learning/1.%20Intro%20(CartPole).ipynb)
2. [Q-Learning](6.%20Deep%20Reinforcement%20Learning/2.%20CartPole.ipynb)
3. [MountainCar](6.%20Deep%20Reinforcement%20Learning/3.%20MountainCar.ipynb)
4. [Lunar Lander](6.%20Deep%20Reinforcement%20Learning/4.%20Lunar%20Lander.ipynb)
5. [Timeseries Prediction in a Financial Environment](6.%20Deep%20Reinforcement%20Learning/5.%20Timeseries%20Prediction%20in%20a%20Financial%20Environment.ipynb)
6. [Live Intraday Trading Agent](6.%20Deep%20Reinforcement%20Learning/6.%20Deploying%20a%20Live%20Intraday%20Trading%20Agent.ipynb)

## Machine Learning

- [Clustering, Lloyd's Algorithm & Expectation-Maximization](8.%20Machine%20Learning/Clustering,%20Lloyd's%20Algorithm%20&%20Expectation-Maximization)
    - [k-Means Clustering](8.%20Machine%20Learning/Clustering,%20Lloyd's%20Algorithm%20&%20Expectation-Maximization/k-Means%20Clustering.ipynb)
    - [Gaussian Mixure Models](8.%20Machine%20Learning/Clustering,%20Lloyd's%20Algorithm%20&%20Expectation-Maximization/Gaussian%20Mixure%20Models.ipynb)
- [Principal Component Analysis (PCA)](8.%20Machine%20Learning/Principal%20Component%20Analysis%20(PCA).ipynb)
- [XGBoost](8.%20Machine%20Learning/XGBoost.ipynb)
- [JAX CPU/GPU Performance Comparison](8.%20Machine%20Learning/JAX%20CPU-GPU%20Performance%20Comparison.ipynb)
- [p-values, Z-scores and t-Tests](8.%20Machine%20Learning/p-values,%20Z-scores%20and%20t-Tests.ipynb)
- [Skewed Distributions, Percentiles and Box Plots](8.%20Machine%20Learning/Skewed%20Distributions,%20Percentiles%20and%20Box%20Plots.ipynb)

## Environment Configuration

To install the conda environment, run

```bash
conda env create -f environment.yml
```

To activate the conda environment, run

```bash
conda activate qfc
```

Feel free to clone, point out any weaknesses/errors, and suggest improvements.

Thank you for exploring this repository. If you have any questions or suggestions, feel free to open an issue or submit a pull request.
