vgg16_trainable.ckpt: VGG_ILSVRC_16_layers.caffemodel
	# for some reason this does not work when called with make
	python caffe_to_tensorflow.py

VGG_ILSVRC_16_layers.caffemodel:
	curl -O http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel

deploy:
	cp vgg16_trainable.ckpt /volatile/DeepLearningData/VGG16Model/vgg16_trainable.ckpt

clean:
	rm -f vgg16_trainable.ckpt


