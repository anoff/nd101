# lessons learned for later lookup

## Multilayer Perceptron

### feature optimization

* try to sketch out significance of features ahead of training
* always make sure that feature gkoptimizations do not bias the network in one direction

### data visualization

* plot frequency of features to identify nodes that might not be necessary
* [t-distributed Stochastic Neighbor Embedding (TSNE)](http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html) to visualize multi-dimensional (e.g. neural network weights)

## CNN

> Convolutional neural networks generate higher order features by applying kernels over one section of a matrix at a time. Stacking convolutions and multiple kernels increases the _feature depth_.

* https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/
  * premiered in 1990 with LeNet
  * excell at image classification/OCR
  * a convolution uses a filter/kernel/feature detector to find patterns in a part of the image
  * the kernel is [moved across the image](https://ujwlkarn.files.wordpress.com/2016/08/giphy.gif?w=748) where it always looks at a small part of it
  * going deep allows the net to find more complex features
  * the CNN learns which features to connect by adjusting weight and biases
* pooling
  * pooling is used to decrease the image size
  * usually used with a kernel stride of 1 (kernel does not reshape the image)
  * typical implementation is using max pooling
  * another patch is moved over the convoluted image and for a given area (pool kernel size) the maximum value is used for the output value
  * basically looking for local maxima in a given area
* network design
  * each convolutional layer should have an activation function
  * convolutions should decrease image size and increase feature depth
  * end with a few fully connected layers (traditional NN incl. activation fns)
  * convolutions detect higher order features (e.g. ears) and fully connected layer (multi layer perceptrons) create _combinations_ of those features for the final classification

## RNN

> Recurrent neural networks are able to adapt their output depending on the sequence in which inputs were given.

* LSTM cells prevent exploding💥/vanishing👻 gradients (would happen if you multiply weights across multiple executions)
* basic LSTMs have gates to **forget**, **update** and **output** the hidden state using combinations of _sigmoid_ and _tanh_
