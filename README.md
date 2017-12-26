# What I've learnt (in reverse chronological order):

- [Built and trained a 2-class classification neural network with a single hidden layer, and implemented forward propagation and backpropagation](Planar+data+classification+with+one+hidden+layer+v4.ipynb)

- [Built a logistic regression classifier to recognize cats](Logistic+Regression+with+a+Neural+Network+mindset+v4.ipynb)

- [Practised Numpy basics (math functions, broadcasting and vectorization)](Python+Basics+With+Numpy+v3.ipynb)

# Notes

## Course 2: Improving Deep Neural Networks

### Setting up your Machine Learning Application

<u>Train/Test/Dev sets</u>
- dev set goals - “hold out cross validation set” - test different algorithms and figure out which works better
- test set goals - given the final classifier figure out how how well it’s doing
- make sure that the dev and test sets come from the same distribution
- not having a test set might be okay

<u>Bias/Variance</u>
- high bias ( —> underfitting) —> high and similar errors with train set and test set
- high variance ( —> overfitting) —> low error with train set and high error with test set
- high bias and high variance —> high error with train set and significantly higher error with test set
- low bias and low variance —> low errors with train and test sets

<i>Note: the error comparisons mentioned above are with respect to the optimal (Bayes) error mentioned</i>
           
<u>Basic recipe for machine learning</u>
Higher bias
- train longer
- bigger network
- find better neural network architecture

High variance
- More data
- Regularization
- find better neural network architecture

### Regularizing NN

Regularization -- add picture of notes

Why does regularization reduce overfitting?

- L~2~ regularization introduces term in cost equation that penalizes weight matrices for being too large
- overfitting —> If lambda is large then W is small then Z is small then tanh is roughly linear then evey layer computes something roughly linear and so it reduces overfitting  

Dropout regularization

- go through each and eliminate some nodes randomly from that layer —> you end with a smaller, diminished network; repeat this for every training example
- Inverted dropout is the most popular implementation


