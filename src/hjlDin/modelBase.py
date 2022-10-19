import os
from abc import ABCMeta, abstractmethod

from tensorflow import keras

try:
    import horovod.tensorflow.keras as hvd
except ModuleNotFoundError as ex:
    print('horovod is not found')


class ModelBuilderBase(metaclass=ABCMeta):
    def __init__(self, ckpt_dir, log_dir):
        self._ckpt_dir = ckpt_dir
        self._log_dir = log_dir



    @abstractmethod
    def build(self):
        raise NotImplementedError()

    def load(self, path=None):
        if (not path) or (not os.path.exists(path)):
            path = os.path.join(self._ckpt_dir, sorted(os.listdir(self._ckpt_dir))[-1])
            print("load路径1：", self._ckpt_dir)
            print("load路径2：", sorted(os.listdir(self._ckpt_dir)))
            print("load路径3：", path)
            path = self._ckpt_dir

        return keras.models.load_model(path)

    def train(self, train_ds, val_ds, epochs, use_hvd, steps):
        ckpt_path = os.path.join(self._ckpt_dir, 'cp-00000001.ckpt')
        ckpt_path = self._ckpt_dir
        print("train路径1：",self._ckpt_dir)
        print("train路径2：",ckpt_path)
        # ckpt_path = os.path.join(self._ckpt_dir, '12345.ckpt')
        ckpt = keras.callbacks.ModelCheckpoint(filepath=ckpt_path, verbose=1, save_best_only=True, period=1)

        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)

        tb = keras.callbacks.TensorBoard(log_dir=self._log_dir, histogram_freq=1, profile_batch=2)

        metrics = [keras.metrics.BinaryAccuracy(), keras.metrics.Precision(),
                   keras.metrics.Recall(), keras.metrics.AUC()]

        callbacks = []
        verbose = 2
        steps_per_epoch = steps

        if use_hvd:
            scaled_lr = 0.001 * hvd.size()

            optimizer = keras.optimizers.Adam(learning_rate=scaled_lr)
            optimizer = hvd.DistributedOptimizer(optimizer=optimizer)

            callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
            callbacks.append(hvd.callbacks.MetricAverageCallback())
            callbacks.append(hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=3, initial_lr=scaled_lr, verbose=1))
            callbacks.append(early_stop)

            steps_per_epoch = steps // hvd.size()

            if hvd.rank() == 0:
                callbacks.append(ckpt)
                callbacks.append(tb)
            else:
                verbose = 0
        else:
            optimizer = keras.optimizers.Adam()
            callbacks.append(ckpt)
            callbacks.append(early_stop)
            callbacks.append(tb)

        model, loss = self.build()
        # single output
        model.output_names[0] = 'predict_score'
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics, experimental_run_tf_function=False)
        history = model.fit(train_ds,
                            validation_data=val_ds,
                            epochs=epochs,
                            steps_per_epoch=steps_per_epoch,
                            callbacks=callbacks,
                            verbose=verbose)

        return model, history