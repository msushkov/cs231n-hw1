import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[0]
  num_train = X.shape[1]
  loss = 0.0
  for i in xrange(num_train):
    curr_pt = X[:, i]
    scores = W.dot(curr_pt)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1.0 # note delta = 1
      if margin > 0:
        loss += margin
        dW[j, :] += curr_pt
        dW[y[i], :] -= curr_pt

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= float(num_train)
  dW /= float(num_train)

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  # Add regularization to the gradient
  dW += reg * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  
  K = W.shape[0]
  N = X.shape[1]
  D = W.shape[1]

  scores = np.dot(W, X)

  y_mat = np.zeros(shape = (K, N))
  y_mat[y, range(N)] = 1

  # matrix of all zeros except for a single wx value in each column that corresponds to the
  # quantity we need to subtract from each row of scores
  correct_wx = np.multiply(y_mat, scores)

  # create a single row of the correct wx_y values for each data point
  sums = np.sum(correct_wx, axis=0) # sum over each column

  margins = scores - sums + 1
  
  # threshold the margins
  result = np.maximum(np.zeros((K, N)), margins)

  # sum over each column
  # (subtract 1 since we overshot our sum by 1 for the j == y term that we didn't ignore)
  result = np.sum(result, axis=0) - 1
  
  # average over all datapoints
  loss = np.sum(result) / float(N)

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  
  # make each entry 1 if it is > 0, 0 otherwise
  margins[margins > 0] = 1
  margins[margins < 0] = 0

  # keep margins mostly the same but for each column, zero out the row corresponding to the
  # correct label
  # (basically change the 1's to 0's, since we are doing w_y*x - w_y*x + 1 for those entries)
  margins[y, range(N)] = 0

  # compute column sums and then set the elements that we zeroed out above to the negative of
  # that sum
  col_sums = np.sum(margins, axis=0)

  margins[y, range(N)] = -1.0 * col_sums

  dW = np.dot(margins, X.T)

  dW /= float(N)

  # Add regularization to the gradient
  dW += reg * W

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
