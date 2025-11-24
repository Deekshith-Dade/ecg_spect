spatioTemporalParams_1D =dict(
integrationMethod = 'concat',
firstLayerParams = dict(in_channels=1,out_channels=32,bias=True,kernel_size=7,maxPoolKernel=4),
lastLayerParams = dict(maxPoolSize=1),
temporalResidualBlockParams1 = dict(
	in_channels=[(32,32),
				 (64,64),
				 (128,128),
				 (256,256)],
	out_channels=[(32,64),
				  (64,128),
				  (128,256),
				  (256,256)],
	kernel_size =[3]*4,
	padding	    =[1]*4,
	numLayers = 4,
	bias = False,
	dropout = [True]*4
	),	
temporalResidualBlockParams2 = dict(
	in_channels=[(32,32),
				 (64,64),
				 (128,128),
				 (256,256)],
	out_channels=[(32,64),
				  (64,128),
				  (128,256),
				  (256,256)],
	kernel_size =[7]*4,
	padding	    =[3]*4,
	numLayers = 4,
	bias = False,
	dropout = [True]*4
	)	
)