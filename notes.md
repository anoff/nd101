# lessons learned for later lookup

## feature optimization

* try to sketch out significance of features ahead of training
* always make sure that feature gkoptimizations do not bias the network in one direction

## data visualization

* plot frequency of features to identify nodes that might not be necessary
* [t-distributed Stochastic Neighbor Embedding (TSNE)](http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html) to visualize multi-dimensional (e.g. neural network weights)
