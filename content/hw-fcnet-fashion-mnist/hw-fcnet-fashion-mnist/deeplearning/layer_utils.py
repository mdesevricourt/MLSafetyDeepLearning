from deeplearning.layers import *


def affine_relu_bn_forward(x, w, b, gamma, beta, bn_params):
    """
    Convenience layer that perorms an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    a, bn_cache = batchnorm_forward(a, gamma, beta, bn_params) 
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, bn_cache, relu_cache)
    return out, cache

def affine_relu_bn_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, bn_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    da, dgamma, dbeta = batchnorm_backward_alt(da, bn_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db, dgamma, dbeta


def affine_relu_do_forward(x, w, b, do_params):
    """
    Convenience layer that perorms an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_relu_forward(x, w, b)
    out, do_cache = dropout_forward(a, do_params) 
    cache = (fc_cache, do_cache)
    return out, cache

def affine_relu_do_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, do_cache = cache
    da = dropout_backward(dout, do_cache)
    dx, dw, db = affine_relu_backward(da, fc_cache)
    return dx, dw, db



def affine_relu_bn_do_forward(x, w, b, gamma, beta, bn_params, do_params):
    """
    Convenience layer that perorms an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    a, bn_cache = batchnorm_forward(a, gamma, beta, bn_params) 
    a, relu_cache = relu_forward(a)
    out, do_cache = dropout_forward(a, do_params)
    cache = (fc_cache, bn_cache, relu_cache, do_cache)
    return out, cache

def affine_relu_bn_do_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, bn_cache, relu_cache, do_cache = cache
    da = dropout_backward(dout, do_cache)
    da = relu_backward(da, relu_cache)
    da, dgamma, dbeta = batchnorm_backward_alt(da, bn_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db, dgamma, dbeta


def conv_relu_forward(x, w, b, conv_param):
    """
    A convenience layer that performs a convolution followed by a ReLU.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    out, relu_cache = relu_forward(a)
    cache = (conv_cache, relu_cache)
    return out, cache


def conv_relu_backward(dout, cache):
    """
    Backward pass for the conv-relu convenience layer.
    """
    conv_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db


