# ISING_NN_PCA
The repository contains programs and models for investigation of estimating phase-transitions in 2D Ising model using Deep Neural Networks and comparison of the results with data compressed by PCA algorithm. The results are part of Master Thesis written on Big Data Analytics on WUST

 - Ising model and Monte-Carlo simulations are written in `C++ 20`. We used `pcg` random number generator.  
 - Data being collected are spin configurations (binary matrices) for different matrix size. 
 - For every dataset, there was created, separately trained and analyzed Deep Neural Network model written in `Python 3.8` with help of `Tensorflow` and `Keras`. 
 - Predictions were repeated on data transformed by dimensionality reduction technique - PCA (from `scikit-learn`) and compared with result on original data.
 - The main goal was to estimate the critical temperature - indicating phase transition in Ising model using Neural Networks and check the impact of dimensionality reduction on estimations.
