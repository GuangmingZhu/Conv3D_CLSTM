## Recurrent layer
class RNNLayer(Layer):
    """
    The :class:`RNNLayer` class is a RNN layer, you can implement vanilla RNN,
    """
    def __init__(
        self,
        layer = None,
        cell_fn = tf.nn.rnn_cell.BasicRNNCell,
        cell_init_args = {},
        n_hidden = 100,
        initializer = tf.random_uniform_initializer(-0.1, 0.1),
        n_steps = 5,
        initial_state = None,
        return_last = False,
        # is_reshape = True,
        return_seq_2d = False,
        name = 'rnn_layer',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs

        print("  tensorlayer:Instantiate RNNLayer %s: n_hidden:%d, n_steps:%d, in_dim:%d %s, cell_fn:%s " % (self.name, n_hidden,
            n_steps, self.inputs.get_shape().ndims, self.inputs.get_shape(), cell_fn.__name__))
        # You can get the dimension by .get_shape() or ._shape, and check the
        # dimension by .with_rank() as follow.
        # self.inputs.get_shape().with_rank(2)
        # self.inputs.get_shape().with_rank(3)

        # Input dimension should be rank 3 [batch_size, n_steps(max), n_features]
        try:
            #self.inputs.get_shape().with_rank(3)
            self.inputs.get_shape().with_rank_at_least(3) #Guangming Zhu
        except:
            raise Exception("RNN : Input dimension should be rank 3 : [batch_size, n_steps, n_features]")


        # is_reshape : boolean (deprecate)
        #     Reshape the inputs to 3 dimension tensor.\n
        #     If input is［batch_size, n_steps, n_features], we do not need to reshape it.\n
        #     If input is [batch_size * n_steps, n_features], we need to reshape it.
        # if is_reshape:
        #     self.inputs = tf.reshape(self.inputs, shape=[-1, n_steps, int(self.inputs._shape[-1])])

        fixed_batch_size = self.inputs.get_shape().with_rank_at_least(1)[0]

        if fixed_batch_size.value:
            batch_size = fixed_batch_size.value
            print("     RNN batch_size (concurrent processes): %d" % batch_size)
        else:
            from tensorflow.python.ops import array_ops
            batch_size = array_ops.shape(self.inputs)[0]
            print("     non specified batch_size, uses a tensor instead.")
        self.batch_size = batch_size
        height = self.inputs.get_shape().as_list()[2] #Guangming Zhu
        width = self.inputs.get_shape().as_list()[3] #Guangming Zhu

        # Simplified version of tensorflow.models.rnn.rnn.py's rnn().
        # This builds an unrolled LSTM for tutorial purposes only.
        # In general, use the rnn() or state_saving_rnn() from rnn.py.
        #
        # The alternative version of the code below is:
        #
        # from tensorflow.models.rnn import rnn
        # inputs = [tf.squeeze(input_, [1])
        #           for input_ in tf.split(1, num_steps, inputs)]
        # outputs, state = rnn.rnn(cell, inputs, initial_state=self._initial_state)
        outputs = []
        self.cell = cell = cell_fn(num_units=n_hidden, **cell_init_args)
        if initial_state is None:
            if self.inputs.get_shape().ndims==3: #Guangming Zhu
                self.initial_state = cell.zero_state(batch_size, dtype=tf.float32)  # 1.2.3
            elif self.inputs.get_shape().ndims==5: #Guangming Zhu
                self.initial_state = cell.zero_state(batch_size, height, width)   #Guangming Zhu
        state = self.initial_state
        # with tf.variable_scope("model", reuse=None, initializer=initializer):
        with tf.variable_scope(name, initializer=initializer) as vs:
            for time_step in range(n_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(self.inputs[:, time_step, :], state)
                outputs.append(cell_output)

            # Retrieve just the RNN variables.
            # rnn_variables = [v for v in tf.all_variables() if v.name.startswith(vs.name)]
            rnn_variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope=vs.name)

        print("     n_params : %d" % (len(rnn_variables)))

        if return_last:
            # 2D Tensor [batch_size, n_hidden]
            self.outputs = outputs[-1]
        else:
            if return_seq_2d:
                # PTB tutorial: stack dense layer after that, or compute the cost from the output
                # 2D Tensor [n_example, n_hidden]
                if self.inputs.get_shape().ndims==3: #Guangming Zhu
                    self.outputs = tf.reshape(tf.concat(1, outputs), [-1, n_hidden])
                elif self.inputs.get_shape().ndims==5: #Guangming Zhu
                    self.outputs = tf.reshape(tf.concat(1, outputs), [-1, height, width, n_hidden]) #Guangming Zhu
            else:
                # <akara>: stack more RNN layer after that
                # 3D Tensor [n_example/n_steps, n_steps, n_hidden]
                if self.inputs.get_shape().ndims==3: #Guangming Zhu
                    self.outputs = tf.reshape(tf.concat(1, outputs), [-1, n_steps, n_hidden])
                elif self.inputs.get_shape().ndims==5: #Guangming Zhu
                    self.outputs = tf.reshape(tf.concat(1, outputs), [-1, n_steps, height, width, n_hidden]) #Guangming Zhu

        self.final_state = state

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        # print(type(self.outputs))
        self.all_layers.extend( [self.outputs] )
        self.all_params.extend( rnn_variables )
