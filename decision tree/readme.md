# Implementation of decision tree with a data set of breast cancer from sklearn

## 1 - we implemented the algorithm by separating training data and leting one line for testing
```python
from sklearn.datasets import load_breast_cancer

ds = load_breast_cancer()

column_names = ds['feature_names']

training_data = ds['data'][:-1]
target = ds['target'][:-1]

testing_item = ds['data'][-1:]
testing_target = ds['target'][-1:]
```
## 2 - the selection of the root
between all the features that we have `worst raduis` is the most influencial feature that will take the role of the root, with entropy of 0.953.

## 3 - trying different variation of depth
after trying multiple variation of depth equal 1, 2 and None we can conclude that:
- if we put 1 as max depth we can't take advantage of all the features that have and mostly we will got an underfitting.
- if we put 2 as max depth the tree will be more complex but it will has more variance between feature, that probably is considered the best case.
- if we put None as max depth we will have a very large depth depending on the number of features and we will also got overfitting because the tree will be too specific to the training data.

## 4 - trying different criterion
- by changing the parameter criterion of the Decision tree class we are changnig the method on which the model is doing feature selection, and when we have done that we noticed a big different on the shape of the tree with depth of None, but in depth of one or two they are similar.
- we also notice that gini is fuster that entropy, it maybe because gini is computationaly less expansive but at the end they give slightly similar results.