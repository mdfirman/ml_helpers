import theano
import lasagne


class ReverseGradient(theano.gof.Op):
    '''
    Theano node which reverses and scales the gradient

    Taken from this question:
    http://stackoverflow.com/a/33889861/279858

    See also this tutorial:
    http://deeplearning.net/software/theano/extending/op.html
    '''
    view_map = {0: [0]}

    __props__ = ('hp_lambda',)

    def __init__(self, hp_lambda):
        super(ReverseGradient, self).__init__()
        self.hp_lambda = hp_lambda

    def make_node(self, x):
        return theano.gof.graph.Apply(self, [x], [x.type.make_variable()])

    def perform(self, node, inputs, output_storage):
        xin, = inputs
        xout, = output_storage
        xout[0] = xin

    def grad(self, input, output_gradients):
        return [-self.hp_lambda * output_gradients[0]]


class ReverseGradientLayer(lasagne.layers.Layer):
    '''
    Lasagne layer which reverses and scales the gradient, using the theano
    node defined above.

    See this tutorial:
    http://lasagne.readthedocs.org/en/latest/user/custom_layers.html
    '''
    def __init__(self, incoming, hp_lambda, **kwargs):
        super(ReverseGradientLayer, self).__init__(incoming, **kwargs)
        self.hp_lambda = hp_lambda

    def get_output_for(self, inp, **kwargs):
        r = ReverseGradient(self.hp_lambda)
        return r(inp)
