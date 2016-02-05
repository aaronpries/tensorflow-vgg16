fgo16.tfmodel: VGG_ILSVRC_16_layers.caffemodel
	PYTHONPATH=../caffe/python:${PYTHONPATH} python caffe_to_fgo.py

VGG_ILSVRC_16_layers.caffemodel:
	curl -O http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel

clean:
	rm -f fgo16.tfmodel
