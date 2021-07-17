# The role of dimensionality reduction in Neural Network based identification of phase transitions
The repository contains programs and models for investigation of estimating phase-transitions in 2D Ising model using Deep Neural Networks and comparison of the results with data compressed by PCA algorithm. The results are part of Master Thesis written on Big Data Analytics at Wrocław University of Science and Technology (WUST)
In short:
 - Ising model and Monte-Carlo simulations are written in `C++ 20`. We used `pcg` random number generator.  
 - Data being collected are spin configurations (binary matrices) for different matrix size. 
 - For every dataset, a Deep Neural Network model there was created, separately trained and analyzed. ML Models are written in `Python 3.8` with help of `Tensorflow` and `Keras`. 
 - Predictions were repeated on data transformed by dimensionality reduction technique - PCA (from `scikit-learn`) and compared with result on original data.
 - The main goal was to estimate the critical temperature - indicating phase transition in Ising model using Neural Networks and check the impact of dimensionality reduction on estimations.
____
## Phase transitions & Ising Model
Statistical mechanics explains macroscopic behaviour of complex systems from the behaviour of large assemblies of microscopic entities. One of the main topics in this area is the research on `phase transitions`, i.e., the abrupt changes of the system (typically behaved dramatically), due to the change of a macroscopic variable.

The **Ising (or Lenz-Ising) model** is the most popular and most unversal model of analyzing phase transitions. We consider 2-dimensional model, since in 1D there is no phase transitions to observe. In the Ising model we consider L x L lattice (or graph in general) of spins (or atoms/agents etc). By analogy to magnetic moments, every spin can take one of the two possible values: "up" / "down" (e.g. 1 or -1). The lattice with spins set in such a way is called `spin configuration`.

Basically, we observe the change of the order parameter, which in this case is called magnetization, i.e., average spin in single configuration:

![formula](https://latex.codecogs.com/svg.latex?m%20%3D%20%5Cfrac%7B1%7D%7BN%7D%20%5Csum_%7Bi%3D1%7D%5EN%20s_i)

The order parameter is different than zero in low temperatures, that is below the critical point (Curie Temperature) -- which indicates ordered, ferromagnetic phase. If we cross the critical point, i.e., the temperature of the system will be sufficiently high, the order parameter is approximately zero and the system is in paramagnetic (disordered) phase. For Ising model, we assume the critical temperature near T ~ 2.269.

