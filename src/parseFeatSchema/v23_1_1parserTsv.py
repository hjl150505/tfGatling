import math
import os
from collections import defaultdict

import pandas as pd
import tensorflow as tf
from collections import namedtuple
from util import *

FeatType = namedtuple('FeatType', ['CatName', 'NumName'])('Cat', 'Num')
print(FeatType)


class FeatParser(object):
    def __init__(self, rp=None, num_default=0.0, num_dtype=tf.float32,
                 cat_default=0, cat_dtype=tf.int64,mean_stddev_path=None):
        self._rp = rp
        self._defaut_map = defaultdict(lambda: '', {FeatType.CatName: cat_default, FeatType.NumName: num_default})
        self._dtype_map = defaultdict(lambda: tf.int64, {FeatType.CatName: cat_dtype, FeatType.NumName: num_dtype})
        self._mean_stddev_path = mean_stddev_path
        self._feat_props = self.parse_feat_config()

    def update_mean_std_if_available(self, num_feat_props):
        if not self._mean_stddev_path:
            return num_feat_props

        rp = self._mean_stddev_path
        rp = get_file_buffer_by_tfds(rp)
        dtype = {'name': str, 'mean': float, 'stddev': float}
        df = pd.read_csv(rp, sep='\t', dtype=dtype)

        stats = {name: (mean, stddev) for name, mean, stddev in df.values}
        mean_stddev_default = (0.0, 1.0)

        updated_mean_std = {}
        for _, name, size, _, _, normalize, *_ in num_feat_props:
            try:
                if not normalize:
                    continue

                mean = None
                stddev = None

                if size == 1:
                    mean = round(stats.get(name, mean_stddev_default)[0], 4)
                    stddev = round(stats.get(name, mean_stddev_default)[1], 4)
                if size > 1:
                    mean = [round(stats.get('{}_{}'.format(name, i), mean_stddev_default)[0], 4) for i in range(size)]
                    stddev = [round(stats.get('{}_{}'.format(name, i), mean_stddev_default)[1], 4) for i in range(size)]

                vs = [mean] if size == 1 else mean
                if any(math.isnan(x) for x in vs):
                    raise ValueError('mean can not be NaN, will be set to {}'.format(mean_stddev_default[0]))

                vs = [stddev] if size == 1 else stddev
                if any(math.isnan(x) for x in vs):
                    raise ValueError('std can not be NaN, will be set to {}'.format(mean_stddev_default[1]))

                if any(x == 0.0 for x in vs):
                    raise ValueError('std can not be 0.0, will be set to {}'.format(mean_stddev_default[1]))

                # rectify mean stddev
                if size == 1:
                    mean = mean_stddev_default[0] if math.isnan(mean) else mean
                    stddev = mean_stddev_default[1] if math.isnan(stddev) else stddev
                if size > 1:
                    mean = [mean_stddev_default[0] if math.isnan(m) else m for m in mean]
                    stddev = [mean_stddev_default[1] if (math.isnan(s) or s == 0.0) else s for s in stddev]

                updated_mean_std[name] = (mean, stddev)
            except (ValueError, AttributeError) as ex:
                print('[ERROR] update mean stddev value ({}) parse error'.format(name))
                print('[ERROR] >>', ex)

        updated_num_feat_props = [(use, name, size,
                                   updated_mean_std.get(name, (mean, stddev))[0],
                                   updated_mean_std.get(name, (mean, stddev))[1],
                                   normalize, default, dtype)
                                  for use, name, size, mean, stddev, normalize, default, dtype in num_feat_props]

        return updated_num_feat_props

    def parse_feat_config(self):
        if not self._rp:
            return None
        dtype_tsv = {
            # common property
            'use': str, 'name': str, 'type': str, 'size': int,
            # categorical feature property
            'buckets': str, 'weight_fc': str,
            # numerical feature property
            'mean': str, 'stddev': str, 'normalize': str,
        }
        df = pd.read_csv(self._rp, sep='\t', dtype=dtype_tsv, keep_default_na=False)

        feat_props = []
        for use, name, type_, size, buckets, weight_fc, mean, stddev, normalize in df.values:
            use = use == 'TRUE'
            buckets_parsed = None
            mean_parsed = None
            stddev_parsed = None
            normalize = normalize == 'TRUE'

            default = self._defaut_map[type_]
            dtype = self._dtype_map[type_]

            if type_ == FeatType.CatName:
                try:
                    buckets_parsed = int(float(buckets))
                    assert buckets_parsed > 0, 'buckets need > 0'
                except ValueError as ex:
                    print('[ERROR] cagegorical feature({}) parse error'.format(name))
                    print('[ERROR] >>', ex)
            if type_ == FeatType.NumName and normalize:
                try:
                    mean_parsed = float(mean) if size == 1 else [float(x.strip()) for x in mean.split(',')]
                    stddev_parsed = float(stddev) if size == 1 else [float(x.strip()) for x in stddev.split(',')]

                    vs = [mean_parsed] if size == 1 else mean_parsed
                    if any(math.isnan(x) for x in vs):
                        raise ValueError('mean can not be Nan')
                    vs = [stddev_parsed] if size == 1 else stddev_parsed
                    if any(math.isnan(x) for x in vs):
                        raise ValueError('std can not be NaN')
                    if any(x == 0.0 for x in vs):
                        raise ValueError('std can not be 0.0')
                except (ValueError, AttributeError) as ex:
                    print('[ERROR] numerical feature({}) parse error'.format(name))
                    print('[ERROR] >>', ex)
            feat_props.append((use, name, type_, size,
                               buckets_parsed, weight_fc,
                               mean_parsed, stddev_parsed, normalize,
                               default, dtype))

        return feat_props
    def get_num_feat(self):
        if self._feat_props is None:
            return None

        num_feat_props = [(use, name, size, mean, stddev, normalize, default, dtype)
                          for use, name, type_, size, buckets, weight_fc, mean, stddev, normalize, default, dtype
                          in self._feat_props if type_ == FeatType.NumName]

        num_feat_props = self.update_mean_std_if_available(num_feat_props)

        return num_feat_props

    def get_cat_feat(self):
        if self._feat_props is None:
            return None

        return [(use, name, size, buckets, weight_fc, default, dtype)
                for use, name, type_, size, buckets, weight_fc, mean, stddev, normalize, default, dtype
                in self._feat_props if type_ == FeatType.CatName]

if __name__ == '__main__':
    feat_parser = FeatParser(rp=r'feat_br.tsv', mean_stddev_path=r'../tfDataSet/mean_stddev.tsv')
    print('num feat')
    print('\n'.join(str(x) for x in feat_parser.get_num_feat()))
    print('cat feat')
    print('\n'.join(str(x) for x in feat_parser.get_cat_feat()))