# Dataset-Normalizer
This Python class has the purpose of transforming continuous variables in a dataset, no matter their distributions, into standard Gaussian variables. 
Its use is the same as the classes in `sklearn.preprocessing`.

# Getting Started

## Prerequisites
In order to use this Python class you will need to have the following modules installed:
- Scipy
- Numpy 
- StatsModels

## Theoretical Background 
I think it's helpful to understand the theory behind why this transformer (I really don't know if that's the correct name, but that's how I'll refer to it here) works, but feel free to skip this part if you wish. It's very simple though.

Suppose we have a dataset <a href="http://www.codecogs.com/eqnedit.php?latex=\{&space;x_1,&space;x_2,&space;x_3,&space;\dots&space;x_n&space;\}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\{&space;x_1,&space;x_2,&space;x_3,&space;\dots&space;x_n&space;\}" title="\{ x_1, x_2, x_3, \dots x_n \}" /></a> with observations of some continuous random variable of interest <a href="http://www.codecogs.com/eqnedit.php?latex=X" target="_blank"><img src="http://latex.codecogs.com/gif.latex?X" title="X" /></a>, the distribution of which we don't know. 
What we do know, however, is its empirical cumulative density function <a href="http://www.codecogs.com/eqnedit.php?latex=\text{eCDF}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\text{eCDF}" title="\text{eCDF}" /></a>, which acts as a transformation of the variable <a href="http://www.codecogs.com/eqnedit.php?latex=X" target="_blank"><img src="http://latex.codecogs.com/gif.latex?X" title="X" /></a> into the <a href="http://www.codecogs.com/eqnedit.php?latex=(0,&space;1)" target="_blank"><img src="http://latex.codecogs.com/gif.latex?(0,&space;1)" title="(0, 1)" /></a> interval. 
You could consider that interval as closed, but only if you are sure the dataset contains the exact lowest and greatest values <a href="http://www.codecogs.com/eqnedit.php?latex=X" target="_blank"><img src="http://latex.codecogs.com/gif.latex?X" title="X" /></a> can have, which is a big assumption.
Otherwise, it is helpful to be ignorant about your variable and assume it has support on some open interval like <a href="http://www.codecogs.com/eqnedit.php?latex=(0,&space;\infty)" target="_blank"><img src="http://latex.codecogs.com/gif.latex?(0,&space;\infty)" title="(0, \infty)" /></a> or <a href="http://www.codecogs.com/eqnedit.php?latex=(-\infty,&space;\infty)" target="_blank"><img src="http://latex.codecogs.com/gif.latex?(-\infty,&space;\infty)" title="(-\infty, \infty)" /></a>.
With a very lax use of mapping notation, we may define <a href="http://www.codecogs.com/eqnedit.php?latex=\text{eCDF}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\text{eCDF}" title="\text{eCDF}" /></a> as

<a href="http://www.codecogs.com/eqnedit.php?latex=\text{eCDF}&space;:&space;X&space;\mapsto&space;(0,&space;1)" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\text{eCDF}&space;:&space;X&space;\mapsto&space;(0,&space;1)" title="\text{eCDF} : X \mapsto (0, 1)" /></a>

Next, we can look at the cumulative distribution function for a standard Gaussian variable <a href="http://www.codecogs.com/eqnedit.php?latex=\mathcal{N}(0,&space;1)" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\mathcal{N}(0,&space;1)" title="\mathcal{N}(0, 1)" /></a>, <a href="http://www.codecogs.com/eqnedit.php?latex=F" target="_blank"><img src="http://latex.codecogs.com/gif.latex?F" title="F" /></a>. 
Its inverse, <a href="http://www.codecogs.com/eqnedit.php?latex=F^{-1}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?F^{-1}" title="F^{-1}" /></a>, acts as a transformation of the <a href="http://www.codecogs.com/eqnedit.php?latex=(0,1)" target="_blank"><img src="http://latex.codecogs.com/gif.latex?(0,1)" title="(0,1)" /></a> interval into <a href="http://www.codecogs.com/eqnedit.php?latex=\mathcal{N}(0,&space;1)" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\mathcal{N}(0,&space;1)" title="\mathcal{N}(0, 1)" /></a>. 
This allows for the statement that

<a href="http://www.codecogs.com/eqnedit.php?latex=F^{-1}&space;:&space;(0,&space;1)&space;\mapsto&space;\mathcal{N}(0,&space;1)" target="_blank"><img src="http://latex.codecogs.com/gif.latex?F^{-1}&space;:&space;(0,&space;1)&space;\mapsto&space;\mathcal{N}(0,&space;1)" title="F^{-1} : (0, 1) \mapsto \mathcal{N}(0, 1)" /></a>

Thus, we may also define the transformation

<a href="http://www.codecogs.com/eqnedit.php?latex=\text{eCDF}&space;\circ&space;F^{-1}&space;:&space;X&space;\mapsto&space;\mathcal{N}(0,&space;1)" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\text{eCDF}&space;\circ&space;F^{-1}&space;:&space;X&space;\mapsto&space;\mathcal{N}(0,&space;1)" title="\text{eCDF} \circ F^{-1} : X \mapsto \mathcal{N}(0, 1)" /></a> 

and its inverse 

<a href="http://www.codecogs.com/eqnedit.php?latex=\text{eCDF}^{-1}&space;\circ&space;F&space;:&space;\mathcal{N}(0,&space;1)&space;\mapsto&space;X" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\text{eCDF}^{-1}&space;\circ&space;F&space;:&space;\mathcal{N}(0,&space;1)&space;\mapsto&space;X" title="\text{eCDF}^{-1} \circ F : \mathcal{N}(0, 1) \mapsto X" /></a>

When combined, they allow us to go from <a href="http://www.codecogs.com/eqnedit.php?latex=X" target="_blank"><img src="http://latex.codecogs.com/gif.latex?X" title="X" /></a> to a standard Gaussian, a much more convenient variable, and back, no matter how <a href="http://www.codecogs.com/eqnedit.php?latex=X" target="_blank"><img src="http://latex.codecogs.com/gif.latex?X" title="X" /></a> is distributed.

# Class Methods and Example
There are 5 methods in this class, all of which work exactly like the ones in `sklearn.preprocessing`.
Know that it is assumed your data is organized in a Numpy array with shape (*m,n*), where *m* is the sample size and *n* is the number of feautures.

- `fit` : Fits the transformer to the given variable data.
- `transform` : Transforms the given data into a observations of a standard Gaussian variable.
- `fit_transform` : Performs `fit` and `transform` in a single line for ease of use.
- `inverse_transform` : Transforms the standard Gaussian variable into the original variable.
- `_reset`  : Reset the transformer parameters.

### Example
Here a Numpy array of random integer values distributed across different ranges is transformed so that every column consists of data from a standard Gaussian.
```
import numpy as np 
import Normalizer

a = np.empty((100, 5))
for i in range(5):
    a[:,i] = np.random.randint(0, 10*(i+1), size=100)
   
scaler = Normalizer() # initialize the class.
scaler.fit(a) # fit it to the data.
t = scaler.transform(a) # transform the data into a standard Gaussian.
t_inv = scaler.inverse_transform(t) # transform the standard Gaussian back into the orginal data.

# Print mean absolute error between the orginial and doubly transformed data
print np.abs(np.mean(t_inv.flatten() - a.flatten()))
# My result was 0.924379265557.
# Keep in mind that the values possess different ranges.
```

## Things to know
- While `fit` and `transform` are reasonably fast, maybe as fast as the methods present in the `sklearn.preprocessing` module, `inverse_transform` is not. It relies in `brentq` from `scipy.optimize` to numerically compute <a href="http://www.codecogs.com/eqnedit.php?latex=\text{eCDF}^{-1}" target="_blank"><img src="http://latex.codecogs.com/gif.latex?\text{eCDF}^{-1}" title="\text{eCDF}^{-1}" /></a> for every array element, thereby making it rather slow.
- There is already a class in `sklearn.preprocessing` named Normalizer, so I'd recommend changing its name if necessary.

## Contributing
Please feel free to contribute however you'd like.
I taught myself everything I know about computer science and large part of my statistics knowledge, so criticism on how to improve my notation and explanation would be greatly appreciated!

## Authors
- Haniel Campos

## License
This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/HanCamp/Dataset-Normalizer/blob/master/LICENSE) file for details


