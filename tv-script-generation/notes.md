# some notes

* a bigger single layer performs better than two/three stacked layers of equal total size
* softmax yielded worse results as ReLU on fully connected
* however the logits on the fully connected shouldn't have an activation at all to reach `loss < 1`
