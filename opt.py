# LEGO sorter project
# Model hypertuning script
# (c) lego-sorter team, 2022-2023

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import logging
import optuna
import tensorflow as tf
from absl import app, flags

import lib.globals
from lib.model import make_model_params
from lib.image_dataset import load_dataset, split_dataset, augment_dataset

logger = logging.getLogger('lego-sorter')
logging.getLogger(optuna.__name__).setLevel(logging.ERROR)

FLAGS = flags.FLAGS
flags.DEFINE_integer('epoch', default=10, lower_bound=1, short_name='n',
    help='Number of epoch to train model for')
flags.DEFINE_integer('trials', default=100, lower_bound=1, short_name='t',
    help='Number of trials')
flags.DEFINE_boolean('debug', False, help='Set debug logging level')
flags.declare_key_flag('gray')
flags.declare_key_flag('edges')
flags.declare_key_flag('zoom')

def main(_):
    """ Hyperoptimization of the LEGO Sorter model """

    logger.setLevel(logging.DEBUG if FLAGS.debug else logging.INFO)
    image_data = load_dataset()
 
    num_labels = len(image_data.class_names)
    train_data, test_data = split_dataset(image_data)
    aug_data = augment_dataset(train_data)

    def opt_fun(trial: optuna.Trial):
        """ Optimization function """

        params = {
            'num_layers': trial.suggest_int('num_layers', 1, 3),
            'units1': trial.suggest_int('units1', 256, 1024, step=64),
            'dropout1': trial.suggest_float('dropout1', 0.0, 0.5),
            'regularize1': trial.suggest_float('regularize1', 0.0, 0.1),
            'regularize0': trial.suggest_float('regularize0', 0.0, 0.1),
            'optimizer': trial.suggest_categorical('optimizer', ['SGD', 'Adam']),
            'label_smoothing': trial.suggest_float('label_smoothing',  0.0, 0.1),
        }
        for n in range(2, params['num_layers']+1):
            layer_size = trial.suggest_int(f'units{n}', 64, 1024, step=64)
            prev_layer_size = params[f'units{n-1}']
            if layer_size > prev_layer_size:
                # layers must be decreasing in size
                raise optuna.TrialPruned(f'Layer {n} size is greater than previous one ({layer_size}>{prev_layer_size})')

            params.update({
                f'units{n}': layer_size,
                f'dropout{n}': trial.suggest_float(f'dropout{n}', 0.0, 0.5),
                f'regularize{n}': trial.suggest_float(f'regularize{n}', 0.0, 0.1),
            })
        match params['optimizer']:
            case 'Adam':
                params['learning_rate'] = trial.suggest_float('adam_learning_rate', 1e-5, 1e-1, log=True)
            case 'SGD':
                params['learning_rate'] = trial.suggest_float('sgd_learning_rate', 1e-5, 1e-1, log=True)
                params['momentum'] = trial.suggest_float('sgd_momentum', 1e-5, 1e-1, log=True)
        logger.debug(f'Optimization run params: {params}')

        model = make_model_params(num_labels, params)
        history = model.fit(aug_data,
                  epochs=FLAGS.epoch,
                  verbose=int(FLAGS.debug),
                  validation_data=test_data,
                  callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        mode='min',
                        patience=3,
                        verbose=0),
                    # optuna.integration.TFKerasPruningCallback(
                    #     trial=trial,
                    #     monitor='val_categorical_accuracy')
                  ])

        acc = history.history['val_categorical_accuracy'][-1]
        logger.debug(f'Validation accuracy: {acc:.4f}')
        loss = history.history['val_loss'][-1]
        logger.debug(f'Validation loss: {loss:.4f}')

        return loss, acc
    
    study = optuna.create_study(
        study_name='lego-sorter', 
        directions=['minimize', 'maximize'],
        # pruner=optuna.pruners.HyperbandPruner(max_resource=FLAGS.epoch, reduction_factor=3, bootstrap_count=4)
    )
    study.optimize(opt_fun, n_trials=FLAGS.trials, show_progress_bar=True)

    logger.info(f'Number of finished trials: {len(study.trials)}')
    for trial in study.best_trials:
        key_val = '\n'.join([f'\t{key}={value}' for key, value in trial.params.items()])
        logger.info(f'Trial {trial.number}:\n' \
                    f'  Loss: {trial.values[0]:.4f}, accuracy: {trial.values[1]:.4f}\n' \
                    f'  Params:\n{key_val}')

   
if __name__ == '__main__':
    try:
        app.run(main)
    except (SystemExit, KeyboardInterrupt):
        pass
