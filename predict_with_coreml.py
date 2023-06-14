import numpy as np
from ctypes import cdll, c_float, c_char_p, c_void_p, POINTER

# loadModel
encoderObj = cdll.LoadLibrary('./objcWrapper.o')
encoderObj.loadModel.argtypes = [c_char_p]
encoderObj.loadModel.restype = c_void_p

mlmodel_handle = encoderObj.loadModel(b'./coreml/CoremlEncoder.mlmodelc')

# predictWith
# prepare input melSegment and output buffer
melSegment = np.ones((1, 80, 3000), dtype=np.float32)
melSegmentDataPtr = melSegment.ctypes.data_as(POINTER(c_float))

n_state = 384; # tiny=384
output_floats = np.ones((1500, n_state), dtype=np.float32)
output_floats_ptr = output_floats.ctypes.data_as(POINTER(c_float))

encoderObj.predictWith.argtypes = [c_void_p, POINTER(c_float), POINTER(c_float)]
encoderObj.predictWith.restypes = None
encoderObj.predictWith(mlmodel_handle, melSegmentDataPtr, output_floats_ptr)

# it should match
# pytorch output: {'output': array([[[-0.28637695, -0.25561523, ..., -0.10253906]]], dtype=float32)
print(output_floats[0][0], output_floats[0][1], output_floats[1500-1][n_state-1])

# closeModel
encoderObj.closeModel.argtypes = [c_void_p]
encoderObj.closeModel.restypes = None
encoderObj.closeModel(mlmodel_handle)
