from utils.external.network import Network

class InceptionV2(Network):
    def __init__(self, inputs, trainable=True, num_classes=10001):
        self.num_classes=num_classes
        Network.__init__(self, inputs, trainable)

    def setup(self):
        (self.feed('input')
         .conv(7, 7, 64, 2, 2, name='conv1/7x7_s2')
         .max_pool(3, 3, 2, 2, name='pool1/3x3_s2')
         .conv(1, 1, 64, 1, 1, name='conv2/3x3_reduce')
         .conv(3, 3, 192, 1, 1, name='conv2/3x3')
         .max_pool(3, 3, 2, 2, name='pool2/3x3_s2')
         .conv(1, 1, 64, 1, 1, name='inception_3a/1x1'))

        (self.feed('pool2/3x3_s2')
         .conv(1, 1, 64, 1, 1, name='inception_3a/3x3_reduce')
         .conv(3, 3, 64, 1, 1, name='inception_3a/3x3'))

        (self.feed('pool2/3x3_s2')
         .conv(1, 1, 64, 1, 1, name='inception_3a/double_3x3_reduce')
         .conv(3, 3, 96, 1, 1, name='inception_3a/double_3x3_1')
         .conv(3, 3, 96, 1, 1, name='inception_3a/double_3x3_2'))

        (self.feed('pool2/3x3_s2')
         .avg_pool(3, 3, 1, 1, name='inception_3a/pool')
         .conv(1, 1, 32, 1, 1, name='inception_3a/pool_proj'))

        (self.feed('inception_3a/1x1',
                   'inception_3a/3x3',
                   'inception_3a/double_3x3_2',
                   'inception_3a/pool_proj')
         .concat(3, name='inception_3a/output')
         .conv(1, 1, 64, 1, 1, name='inception_3b/1x1'))

        (self.feed('inception_3a/output')
         .conv(1, 1, 64, 1, 1, name='inception_3b/3x3_reduce')
         .conv(3, 3, 96, 1, 1, name='inception_3b/3x3'))

        (self.feed('inception_3a/output')
         .conv(1, 1, 64, 1, 1, name='inception_3b/double_3x3_reduce')
         .conv(3, 3, 96, 1, 1, name='inception_3b/double_3x3_1')
         .conv(3, 3, 96, 1, 1, name='inception_3b/double_3x3_2'))

        (self.feed('inception_3a/output')
         .avg_pool(3, 3, 1, 1, name='inception_3b/pool')
         .conv(1, 1, 64, 1, 1, name='inception_3b/pool_proj'))

        (self.feed('inception_3b/1x1',
                   'inception_3b/3x3',
                   'inception_3b/double_3x3_2',
                   'inception_3b/pool_proj')
         .concat(3, name='inception_3b/output')
         .conv(1, 1, 128, 1, 1, name='inception_3c/3x3_reduce')
         .conv(3, 3, 160, 2, 2, name='inception_3c/3x3'))

        (self.feed('inception_3b/output')
         .conv(1, 1, 64, 1, 1, name='inception_3c/double_3x3_reduce')
         .conv(3, 3, 96, 1, 1, name='inception_3c/double_3x3_1')
         .conv(3, 3, 96, 2, 2, name='inception_3c/double_3x3_2'))

        (self.feed('inception_3b/output')
         .max_pool(3, 3, 2, 2, name='inception_3c/pool'))

        (self.feed('inception_3c/pool',
                   'inception_3c/3x3',
                   'inception_3c/double_3x3_2')
         .concat(3, name='inception_3c/output')
         .conv(1, 1, 224, 1, 1, name='inception_4a/1x1'))

        (self.feed('inception_3c/output')
         .conv(1, 1, 64, 1, 1, name='inception_4a/3x3_reduce')
         .conv(3, 3, 96, 1, 1, name='inception_4a/3x3'))

        (self.feed('inception_3c/output')
         .conv(1, 1, 96, 1, 1, name='inception_4a/double_3x3_reduce')
         .conv(3, 3, 128, 1, 1, name='inception_4a/double_3x3_1')
         .conv(3, 3, 128, 1, 1, name='inception_4a/double_3x3_2'))

        (self.feed('inception_3c/output')
         .avg_pool(3, 3, 1, 1, name='inception_4a/pool')
         .conv(1, 1, 128, 1, 1, name='inception_4a/pool_proj'))

        (self.feed('inception_4a/1x1',
                   'inception_4a/3x3',
                   'inception_4a/double_3x3_2',
                   'inception_4a/pool_proj')
         .concat(3, name='inception_4a/output')
         .conv(1, 1, 192, 1, 1, name='inception_4b/1x1'))

        (self.feed('inception_4a/output')
         .conv(1, 1, 96, 1, 1, name='inception_4b/3x3_reduce')
         .conv(3, 3, 128, 1, 1, name='inception_4b/3x3'))

        (self.feed('inception_4a/output')
         .conv(1, 1, 96, 1, 1, name='inception_4b/double_3x3_reduce')
         .conv(3, 3, 128, 1, 1, name='inception_4b/double_3x3_1')
         .conv(3, 3, 128, 1, 1, name='inception_4b/double_3x3_2'))

        (self.feed('inception_4a/output')
         .avg_pool(3, 3, 1, 1, name='inception_4b/pool')
         .conv(1, 1, 128, 1, 1, name='inception_4b/pool_proj'))

        (self.feed('inception_4b/1x1',
                   'inception_4b/3x3',
                   'inception_4b/double_3x3_2',
                   'inception_4b/pool_proj')
         .concat(3, name='inception_4b/output')
         .conv(1, 1, 160, 1, 1, name='inception_4c/1x1'))

        (self.feed('inception_4b/output')
         .conv(1, 1, 128, 1, 1, name='inception_4c/3x3_reduce')
         .conv(3, 3, 160, 1, 1, name='inception_4c/3x3'))

        (self.feed('inception_4b/output')
         .conv(1, 1, 128, 1, 1, name='inception_4c/double_3x3_reduce')
         .conv(3, 3, 160, 1, 1, name='inception_4c/double_3x3_1')
         .conv(3, 3, 160, 1, 1, name='inception_4c/double_3x3_2'))

        (self.feed('inception_4b/output')
         .avg_pool(3, 3, 1, 1, name='inception_4c/pool')
         .conv(1, 1, 128, 1, 1, name='inception_4c/pool_proj'))

        (self.feed('inception_4c/1x1',
                   'inception_4c/3x3',
                   'inception_4c/double_3x3_2',
                   'inception_4c/pool_proj')
         .concat(3, name='inception_4c/output')
         .conv(1, 1, 96, 1, 1, name='inception_4d/1x1'))

        (self.feed('inception_4c/output')
         .conv(1, 1, 128, 1, 1, name='inception_4d/3x3_reduce')
         .conv(3, 3, 192, 1, 1, name='inception_4d/3x3'))

        (self.feed('inception_4c/output')
         .conv(1, 1, 160, 1, 1, name='inception_4d/double_3x3_reduce')
         .conv(3, 3, 192, 1, 1, name='inception_4d/double_3x3_1')
         .conv(3, 3, 192, 1, 1, name='inception_4d/double_3x3_2'))

        (self.feed('inception_4c/output')
         .avg_pool(3, 3, 1, 1, name='inception_4d/pool')
         .conv(1, 1, 128, 1, 1, name='inception_4d/pool_proj'))

        (self.feed('inception_4d/1x1',
                   'inception_4d/3x3',
                   'inception_4d/double_3x3_2',
                   'inception_4d/pool_proj')
         .concat(3, name='inception_4d/output')
         .conv(1, 1, 128, 1, 1, name='inception_4e/3x3_reduce')
         .conv(3, 3, 192, 2, 2, name='inception_4e/3x3'))

        (self.feed('inception_4d/output')
         .conv(1, 1, 192, 1, 1, name='inception_4e/double_3x3_reduce')
         .conv(3, 3, 256, 1, 1, name='inception_4e/double_3x3_1')
         .conv(3, 3, 256, 2, 2, name='inception_4e/double_3x3_2'))

        (self.feed('inception_4d/output')
         .max_pool(3, 3, 2, 2, name='inception_4e/pool'))

        (self.feed('inception_4e/pool',
                   'inception_4e/3x3',
                   'inception_4e/double_3x3_2')
         .concat(3, name='inception_4e/output')
         .conv(1, 1, 352, 1, 1, name='inception_5a/1x1'))

        (self.feed('inception_4e/output')
         .conv(1, 1, 192, 1, 1, name='inception_5a/3x3_reduce')
         .conv(3, 3, 320, 1, 1, name='inception_5a/3x3'))

        (self.feed('inception_4e/output')
         .conv(1, 1, 160, 1, 1, name='inception_5a/double_3x3_reduce')
         .conv(3, 3, 224, 1, 1, name='inception_5a/double_3x3_1')
         .conv(3, 3, 224, 1, 1, name='inception_5a/double_3x3_2'))

        (self.feed('inception_4e/output')
         .avg_pool(3, 3, 1, 1, name='inception_5a/pool')
         .conv(1, 1, 128, 1, 1, name='inception_5a/pool_proj'))

        (self.feed('inception_5a/1x1',
                   'inception_5a/3x3',
                   'inception_5a/double_3x3_2',
                   'inception_5a/pool_proj')
         .concat(3, name='inception_5a/output')
         .conv(1, 1, 352, 1, 1, name='inception_5b/1x1'))

        (self.feed('inception_5a/output')
         .conv(1, 1, 192, 1, 1, name='inception_5b/3x3_reduce')
         .conv(3, 3, 320, 1, 1, name='inception_5b/3x3'))

        (self.feed('inception_5a/output')
         .conv(1, 1, 192, 1, 1, name='inception_5b/double_3x3_reduce')
         .conv(3, 3, 224, 1, 1, name='inception_5b/double_3x3_1')
         .conv(3, 3, 224, 1, 1, name='inception_5b/double_3x3_2'))

        (self.feed('inception_5a/output')
         .max_pool(3, 3, 1, 1, name='inception_5b/pool')
         .conv(1, 1, 128, 1, 1, name='inception_5b/pool_proj'))

        (self.feed('inception_5b/1x1',
                   'inception_5b/3x3',
                   'inception_5b/double_3x3_2',
                   'inception_5b/pool_proj')
         .concat(3, name='inception_5b/output')
         .avg_pool(7, 7, 1, 1, padding='VALID', name='global_pool')
         .fc(self.num_classes, relu=False, name='fc'))