from TinyNet.modules.conv import ConvLayer


class LinearLayer(ConvLayer):
    def __init__(self,linear_config,input_shape,output_shape,misc_config):

        linear_args = [
            'in_features','out_features','bias','activation_func'
        ]

        self.in_features = linear_config['in_features']
        self.out_features = linear_config['out_features']
        # self.bias = bias
        # self.act_function = 'relu'

        # HWC

        conv_config = {
            'in_channels':input_shape[2],'out_channels':self.out_features,'kernel_size':input_shape[0:2],
            'stride':(1,1),'padding':(0,0),'groups':1,'bias':linear_config['bias'],
            'activation_function':linear_config['activation_function']
        }

        # input_shape = (1,1,self.in_features)
        # output_shape = (1,1,self.out_features)

        super(LinearLayer, self).__init__(conv_config,input_shape,output_shape,misc_config)






