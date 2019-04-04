import numpy

from chainer.backends import cuda
from chainer import function_node
from chainer.utils import type_check
import six
import cupy



def _prepend_const(narray, pad_amount, value, axis=-1):
    if pad_amount == 0:
        return narray
    padshape = tuple(x if i != axis else pad_amount
                     for i, x in enumerate(narray.shape))
    return cupy.concatenate((cupy.full(padshape, value, narray.dtype),
                             narray), axis=axis)


def _append_const(narray, pad_amount, value, axis=-1):
    if pad_amount == 0:
        return narray
    padshape = tuple(x if i != axis else pad_amount
                     for i, x in enumerate(narray.shape))
    return cupy.concatenate((narray,
                             cupy.full(padshape, value, narray.dtype)),
                            axis=axis)

def _prepend_reflect(narray, pad_amount, value, axis=-1):
    if pad_amount == 0:
        return narray
    padshape = tuple(x if i != axis else pad_amount
                     for i, x in enumerate(narray.shape))
    return cupy.concatenate((cupy.full(padshape, value, narray.dtype),
                             narray), axis=axis)


def _append_reflect(narray, pad_amount, value, axis=-1):
    if pad_amount == 0:
        return narray
    padshape = tuple(x if i != axis else pad_amount
                     for i, x in enumerate(narray.shape))
    return cupy.concatenate((narray,
                             cupy.full(padshape, value, narray.dtype)),
                            axis=axis)


def _normalize_shape(ndarray, shape, cast_to_int=True):
    ndims = ndarray.ndim
    if shape is None:
        return ((None, None), ) * ndims
    ndshape = numpy.asarray(shape)
    # 0410
    #print(ndshape, ndims, ndshape.size, ndshape.ndim, ndshape.shape)
    if ndshape.size == 1:#1
        ndshape = numpy.repeat(ndshape, 2)
    if ndshape.ndim == 1:#0
        ndshape = numpy.tile(ndshape, (ndims, 1))
    if ndshape.shape != (ndims, 2):#0
        message = 'Unable to create correctly shaped tuple from %s' % shape
        raise ValueError(message)
    if cast_to_int:#1
        ndshape = numpy.rint(ndshape).astype(int)
    return tuple(tuple(axis) for axis in ndshape)

def _normalize_shape2(ndarray, shape, cast_to_int=True):
    ndims = ndarray.ndim
    if shape is None:
        return ((None, None), ) * ndims
    ndshape = numpy.asarray(shape)
    print(ndshape, ndims, ndshape.size, ndshape.ndim)
    if ndshape.size == 1:
        ndshape = numpy.repeat(ndshape, 2)
    if ndshape.ndim == 1:
        ndshape = numpy.tile(ndshape, (ndims, 1))
    if ndshape.shape != (ndims, 2):
        message = 'Unable to create correctly shaped tuple from %s' % shape
        raise ValueError(message)
    if cast_to_int:
        ndshape = numpy.rint(ndshape).astype(int)
    return tuple(tuple(axis) for axis in ndshape)


def _validate_lengths(narray, number_elements):
    shape = _normalize_shape(narray, number_elements)
    for axis_shape in shape:
        axis_shape = [1 if x is None else x for x in axis_shape]
        axis_shape = [1 if x >= 0 else -1 for x in axis_shape]
        if axis_shape[0] < 0 or axis_shape[1] < 0:
            message = '%s cannot contain negative values.' % number_elements
            raise ValueError(message)
    return shape


def mypad(array, pad_width, mode, **keywords):
    """Returns padded array. You can specify the padded widths and values.

    This function currently supports only ``mode=constant`` .

    Args:
        array (array-like): Input array of rank N.
        pad_width (int or array-like): Number of values padded
            to the edges of each axis.
            ((before_1, after_1), ... (before_N, after_N)) uniquely pad widths
            for each axis.
            ((before, after),) yields same before and after pad for each axis.
            (pad,) or int is a shortcut for before = after = pad width for all
            axes.
            You cannot specify ``cupy.ndarray`` .
        mode (str):
            'constant'
                Pads with a constant values.
        constant_values (int or array-like): Used in
            ``constant``.
            The values are padded for each axis.
            ((before_1, after_1), ... (before_N, after_N)) uniquely pad
            constants for each axis.
            ((before, after),) yields same before and after constants for each
            axis.
            (constant,) or int is a shortcut for before = after = constant for
            all axes.
            Default is 0. You cannot specify ``cupy.ndarray`` .

    Returns:
        cupy.ndarray:
        Padded array of rank equal to ``array`` with shape increased according
        to ``pad_width`` .

    .. seealso:: :func:`numpy.pad`

    """
    if not numpy.asarray(pad_width).dtype.kind == 'i':
        raise TypeError('pad_width must be of integral type.')
    narray = cupy.array(array)
    pad_width = _validate_lengths(narray, pad_width)
    newmatrix = cupy.zeros((narray.shape[0],narray.shape[1],narray.shape[2]+2,narray.shape[3]+2), dtype=narray.dtype)
    newmatrix[:,:,1:-1,1:-1]=narray[:,:]
    newmatrix[:,:,0,1:-1]=narray[:,:,1,:]
    newmatrix[:,:,-1,1:-1]=narray[:,:,-2,:]
    newmatrix[:,:,1:-1,0]=narray[:,:,:,1]
    newmatrix[:,:,1:-1,-1]=narray[:,:,:,-2]
    newmatrix[:,:,0,0]=narray[:,:,1,1]
    newmatrix[:,:,-1,0]=narray[:,:,-2,1]
    newmatrix[:,:,0,-1]=narray[:,:,1,-2]
    newmatrix[:,:,-1,-1]=narray[:,:,-2,-2]
    return newmatrix

class Pad(function_node.FunctionNode):

    """Padding of an array."""

    def __init__(self, pad_width, mode, **keywords):
        self.mode = mode
        self.keywords = keywords
        self.pad_width = pad_width
        self.pad_bw = numpy.asarray(pad_width)
        if self.pad_bw.size == 1:
            self.pad_bw = numpy.repeat(self.pad_bw, 2)

    def check_type_forward(self, in_types):
        # Depending on the arguments, pad_width and keywords, the input value
        # may be inappropriate. In that case, numpy.pad or cupy.pad will raise
        # errors, so that only check the size and the dtype in this function.
        type_check.expect(in_types.size() == 1)
        x_type = in_types[0]
        type_check.expect(x_type.dtype.kind == 'f')

    def forward(self, inputs):
        #print(inputs[0].shape)
        xp = cuda.get_array_module(*inputs)
        return mypad(inputs[0], self.pad_width, mode=self.mode,
                      **self.keywords),

    def backward(self, inputs, grad_outputs):
        gy, = grad_outputs
        #print(gy.shape)
        #print('input',self.inputs[0].shape)
        in_shape = self.inputs[0].shape
        if self.pad_bw.ndim == 1:
            self.pad_bw = numpy.tile(self.pad_bw, (len(in_shape), 1))
        #input_idxs = tuple(
        #    slice(p[0], p[0] + dim) for dim, p in zip(in_shape, self.pad_bw))
        input_idxs = tuple((
            slice(0, 0 + in_shape[0]),slice(0, 0 + in_shape[1]),slice(1, 1 + in_shape[2]),slice(1, 1 + in_shape[3])))
        #print(in_shape)
        #print(input_idxs)
        #print(gy[input_idxs].shape)
        return gy[input_idxs],


def pad(x, pad_width, mode, **keywords):
    """Pad an input variable.
    Args:
        x (chainer.Variable or :class:`numpy.ndarray` or cupy.ndarray):
            Input data.
        pad_width (int or array-like):
            Number of values padded to the edges of each axis.
        mode (str):
            Specifies how the function fills the periphery of the array.
            The mode is passed to :func:`numpy.pad` or :func:`cupy.pad`.
            If it is ``'constant'``, the input is padded by a constant value
            specified by ``constant_values``.
        constant_values (int or array-like):
            Constant values to fill the periphery in the ``'constant'`` mode.
    Returns:
        ~chainer.Variable: Output variable.
    """
    return Pad(pad_width, mode, **keywords).apply((x,))[0]
