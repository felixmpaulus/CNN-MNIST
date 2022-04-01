## Building a basic NN from scratch. 

I am building a (convolutional) neural network from scratch to get a better understanding of the underlying fundamentals.
### Ressources

Using 
- https://zerowithdot.com/mlp-backpropagation/ (this one in particular)
- https://blog.yani.ai/backpropagation/ (this one in particular)
- https://brilliant.org/wiki/backpropagation/ (this one in particular)
- https://hmkcode.com/ai/backpropagation-step-by-step/
- https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
- http://neuralnetworksanddeeplearning.com/chap2.html
- https://home.agh.edu.pl/~vlsi/AI/backp_t_en/backprop.html
as a guide.

Debugging:
- https://stats.stackexchange.com/questions/352036/what-should-i-do-when-my-neural-network-doesnt-learn
- https://lecture-demo.ira.uka.de/neural-network-demo/?preset=Binary%20Classifier%20for%20XOR

determining network size
- https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw


### Architecture

- the network uses classic backpropagation and biases for each neuron.
- the final error is calculated using the mean square error formular.
- so far sigmoid, ReLU und leaky ReLU are implemented as activation functions.

- Ideas:
    - different activation for hidden and output neurons
    - implement momentum (see this answer)[https://stackoverflow.com/a/9372629/8179042]
    - different weight intervals for bias weights and normal weights

### XOR
- the network consists of 2 Input, 1 hidden Layer with 2 Neurons and one Output.

**Examining the network**
- non-convergence rate is examined by the following parameters
    - learning rate
        - 0.5
        - 0.3
        - 0.1
        - 0.01
    - activation function
        - sigmoid
        - ReLU
        - leaky ReLU
    - intervall of initial weights
        - [0, 1]
        - [-0.5, 0.5]
        - [0, 0.5]
        - [0.5, 1]

**Results**
- the percentage of non-convergence is shown below. 30 samples were taken with each settings, each network was trained 120 000 times. 

- sigmoid:
    <img src="XOR examination/sigmoid.png" width="800"/>

- ReLU:
    <img src="XOR examination/ReLU.png" width="800"/>

- leaky ReLU (0.01x):
    <img src="XOR examination/leaky ReLU 0.01.png" width="800"/>

- leaky ReLU (0.001x):
    <img src="XOR examination/leaky ReLU 0.001.png" width="800"/>

The settings that resulted in the lowest non-convergence rate were the following:
- ReLU, weights in [0, 1], learning rate 0.01 at **7% non-convergence**
- ReLU, weights in [0.5, 1], learning rate 0.01 at **7% non-convergence**
- leaky ReLU with 0.01x, weights in [0.5, 1], learning rate 0.01 at **7% non-convergence**
- leaky ReLU with 0.001x, weights in [0, 1], learning rate 0.01 at **7% non-convergence**

**Loss during training**
- the following graphs show the decreasing loss of the previously mentioned settings.
- ReLU, weights in [0, 1], learning rate 0.01:
    <img src="XOR examination/ReLU loss 1.png" width="800"/>

- ReLU, weights in [0.5, 1], learning rate 0.01:
    <img src="XOR examination/ReLU loss 2.png" width="800"/>

- leaky ReLU with 0.01x, weights in [0.5, 1], learning rate 0.01:
    <img src="XOR examination/lReLU loss 1.png" width="800"/>

- leaky ReLU with 0.001x, weights in [0, 1], learning rate 0.01:
    <img src="XOR examination/lReLU loss 2.png" width="800"/>






### Visualization (not of this project):
- http://playground.tensorflow.org/

### other stuff
- kaggle.com 