# Conv3D_CLSTM

## Prerequisites

1) Tensorflow-0.11 <br/>
2) Tensorlayer (commit ba30379f1b86f930d6e86e1c8db49cbd2d9aa314) <br/> 
   The original Tensorlayer does not support the convolutional LSTM, so the class RNNLayer in tensorlayer/layer.py needs to be modified according to the tensorlayer-rnnlayer.py. <br/>
