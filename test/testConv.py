from TinyNet.modules import conv
from TinyDSL.HwResource.config import core_config

conv_config = {
    'in_channels':32,'out_channels':64,'kernel_size':(3,3),'stride':(1,1),
    'padding':(2,2),'group':1,'bias':True,'activation_func':'relu'
}

misc_config = {
    'in_act_pad_shape':(28,28,32),'out_act_shape':(28,28,64),'meu_layout':(3,2),
    'mat_shape':(288,64),'mat_bitwidth':1,'act_bitwidth':1,'core_config':core_config,'posi':[slice(0,288),slice(0,64)]
}

cc = conv.ConvCore(conv_config=conv_config,misc_config=misc_config)
cc.compute(1,1)
cc.core.inst_buffer.dump_binary()
print("in deconstruction")