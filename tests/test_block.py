from keras_resnet import blocks

class TestTimeDistributedBottlneck2D:
    def test_constructor(self, num_filters):
        block = blocks.time_distributed_bottleneck_2d(num_filters, block=0)

        assert callable(block.f)
