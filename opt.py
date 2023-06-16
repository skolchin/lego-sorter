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
from lib.model import make_model_with_params
from lib.image_dataset import load_dataset, split_dataset, augment_dataset

logger = logging.getLogger('lego-sorter')

FLAGS = flags.FLAGS
flags.DEFINE_integer('epoch', default=10, lower_bound=1, short_name='n',
    help='Number of epoch to train model for')
flags.DEFINE_integer('trials', default=100, lower_bound=1, short_name='t',
    help='Number of trials')
flags.DEFINE_enum('target', 'both', ['loss', 'accuracy', 'both'], short_name='r',
    help='Optimization target')
flags.DEFINE_boolean('debug', False, help='Set debug logging level')
flags.declare_key_flag('gray')
flags.declare_key_flag('edges')
flags.declare_key_flag('zoom')

def suggest_params(trial: optuna.Trial) -> dict:
    params = {
        'num_layers': trial.suggest_int('num_layers', 2, 4),
        'apply_gap': trial.suggest_int('apply_gap', 0, 1),
        'units1': trial.suggest_int('units1', 256, 1024, step=64),
        'dropout1': trial.suggest_float('dropout1', 0.0, 0.5),
        'regularize1': trial.suggest_float('regularize1', 0.0, 0.1),
        'regularize0': trial.suggest_float('regularize0', 0.0, 0.1),
        'optimizer': trial.suggest_categorical('optimizer', ['SGD', 'Adam', 'RMSProp']),
        'label_smoothing': trial.suggest_float('label_smoothing',  0.0, 0.1),
    }
    for n in range(2, params['num_layers']+1):
        layer_size = trial.suggest_int(f'units{n}', 64, 1024, step=64)
        prev_layer_size = params[f'units{n-1}']
        # if layer_size > prev_layer_size:
        #     # layers must be decreasing in size
        #     raise optuna.TrialPruned(f'Layer {n} size is greater than previous one ({layer_size}>{prev_layer_size})')

        params.update({
            f'units{n}': layer_size,
            f'dropout{n}': trial.suggest_float(f'dropout{n}', 0.0, 0.5),
            f'regularize{n}': trial.suggest_float(f'regularize{n}', 0.0, 0.1),
        })
    match params['optimizer']:
        case 'Adam':
            params['learning_rate'] = trial.suggest_float('adam_learning_rate', 1e-4, 1e-1, log=True)

        case 'SGD':
            params['learning_rate'] = trial.suggest_float('sgd_learning_rate', 1e-4, 1e-1, log=True)
            params['momentum'] = trial.suggest_float('sgd_momentum', 1e-4, 1e-1, log=True)

        case 'RMSProp':
            params['learning_rate'] = trial.suggest_float('rms_learning_rate', 1e-4, 1e-1, log=True)
            params['momentum'] = trial.suggest_float('rms_momentum', 1e-4, 1e-1, log=True)

    logger.debug(f'Optimization run params: {params}')
    return params

def get_metrics(history):
    loss = history.history['val_loss'][-1]
    logger.debug(f'Validation loss: {loss:.4f}')

    acc = history.history['val_categorical_accuracy'][-1]
    logger.debug(f'Validation accuracy: {acc:.4f}')
    return loss, acc

def print_metric(study: optuna.Study, metric: str):
    logger.info(f'Number of finished trials: {len(study.trials)}')
    key_val = '\n'.join([f'\t{key}={value}' for key, value in study.best_trial.params.items()])
    logger.info(f'Best trial:\n' \
                f'  {metric}: {study.best_trial.value:.4f}\n' \
                f'  Params:\n{key_val}')

def print_metrics(study: optuna.Study):
    logger.info(f'Number of finished trials: {len(study.trials)}')
    for trial in study.best_trials:
        key_val = '\n'.join([f'\t{key}={value}' for key, value in trial.params.items()])
        logger.info(f'Trial {trial.number}:\n' \
                    f'  Loss: {trial.values[0]:.4f}, accuracy: {trial.values[1]:.4f}\n' \
                    f'  Params:\n{key_val}')

def main(_):
    """ Hyperoptimization of the LEGO Sorter model """

    # Data prep
    logger.setLevel(logging.DEBUG if FLAGS.debug else logging.INFO)
    logging.getLogger(optuna.__name__).setLevel(logging.INFO if FLAGS.debug else logging.ERROR)
    image_data = load_dataset()
 
    num_labels = len(image_data.class_names)
    train_data, test_data = split_dataset(image_data)
    aug_data = augment_dataset(train_data)

    # Family of optimization funcs for different targets
    def opt_fun_loss(trial: optuna.Trial):
        """ Optimization function for loss target"""

        params = suggest_params(trial)
        model = make_model_with_params(num_labels, params)
        callbacks = [optuna.integration.TFKerasPruningCallback(trial=trial, monitor='val_loss')]
        history = model.fit(aug_data,
                  epochs=FLAGS.epoch,
                  verbose=int(FLAGS.debug),
                  validation_data=test_data,
                  callbacks=callbacks)

        loss, acc = get_metrics(history)
        return loss

    def opt_fun_accuracy(trial: optuna.Trial):
        """ Optimization function for accuracy target"""

        params = suggest_params(trial)
        model = make_model_with_params(num_labels, params)
        callbacks = [optuna.integration.TFKerasPruningCallback(trial=trial, monitor='val_categorical_accuracy')]
        history = model.fit(aug_data,
                  epochs=FLAGS.epoch,
                  verbose=int(FLAGS.debug),
                  validation_data=test_data,
                  callbacks=callbacks)

        loss, acc = get_metrics(history)
        return acc

    def opt_fun_all(trial: optuna.Trial):
        """ Optimization function for accuracy and loss targets """

        params = suggest_params(trial)
        model = make_model_with_params(num_labels, params)
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=3, verbose=0)]
        history = model.fit(aug_data,
                  epochs=FLAGS.epoch,
                  verbose=int(FLAGS.debug),
                  validation_data=test_data,
                  callbacks=callbacks)

        loss, acc = get_metrics(history)
        return loss, acc

    # Main optimization routine
    match FLAGS.target:
        case 'loss':
            logger.info('Optimiziting for minimal loss')
            study = optuna.create_study(study_name='lego-sorter-loss', direction='minimize',
                pruner=optuna.pruners.MedianPruner())
            study.optimize(opt_fun_loss, n_trials=FLAGS.trials, show_progress_bar=True)
            print_metric(study, 'loss')

        case 'accuracy':
            logger.info('Optimiziting for maximum accuracy')
            study = optuna.create_study(study_name='lego-sorter-accuracy', direction='maximize',
                pruner=optuna.pruners.MedianPruner())
            study.optimize(opt_fun_accuracy, n_trials=FLAGS.trials, show_progress_bar=True)
            print_metric(study, 'accuracy')
 
        case _:
            logger.info('Targeting both loss and accuracy')
            study = optuna.create_study(study_name='lego-sorter', directions=['minimize', 'maximize'])
            study.optimize(opt_fun_all, n_trials=FLAGS.trials, show_progress_bar=True)
            print_metrics(study)

   
if __name__ == '__main__':
    try:
        app.run(main)
    except (SystemExit, KeyboardInterrupt):
        pass
