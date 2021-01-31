#!/usr/bin/env python3
"""
Time-series adaptation
"""
import os

# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
import sys
import time
import tensorflow as tf

from absl import app
from absl import flags

import models
import methods
import file_utils
import load_datasets

from datasets import datasets
from metrics import Metrics
from checkpoints import CheckpointManager
from gpu_memory import set_gpu_memory

print (methods.list_methods())
print (models.list_models())
print (datasets.list_datasets())
FLAGS = flags.FLAGS

flags.DEFINE_string("modeldir", "example-models", "Directory for saving model files")
flags.DEFINE_string("logdir", "example-logs", "Directory for saving log files")
flags.DEFINE_enum("method", "yndaws", methods.list_methods(), "What method of domain adaptation to perform (or none)")
flags.DEFINE_enum("model", "ynfcn", models.list_models(), "What model to use (note: ignored for vrada/rdann methods)")
flags.DEFINE_enum("dataset", "ucihar", datasets.list_datasets(), "What dataset to use (e.g. \"ucihar\")")
flags.DEFINE_string("sources", "14", "Which source domains to use (e.g. \"1,2,3\")")
flags.DEFINE_string("target", "19", "What target domain to use (e.g. \"4\", can be blank for no target)")
flags.DEFINE_string("uid", "0", "A unique ID saved in the log/model folder names to avoid conflicts")
flags.DEFINE_integer("ensemble", 1, "Number of models in the ensemble, 1 = no ensemble")
flags.DEFINE_integer("steps", 30000, "Number of training steps to run")
flags.DEFINE_float("gpumem", 2000, "GPU memory to let TensorFlow use, in MiB (0 for all)")
flags.DEFINE_integer("model_steps", 0, "Save the model every so many steps (0 for only when log_val_steps)")
flags.DEFINE_integer("log_train_steps", 500, "Log training information every so many steps (0 for never)")
flags.DEFINE_integer("log_val_steps", 5, "Log validation information every so many steps (also saves model, 0 for only at end)")
flags.DEFINE_integer("log_plots_steps", 0, "Log plots every so many steps (0 for never)")
flags.DEFINE_boolean("test", False, "Use real test set for evaluation rather than validation set")
flags.DEFINE_boolean("subdir", True, "Save models/logs in subdirectory of prefix")
flags.DEFINE_boolean("debug", False, "Start new log/model/images rather than continuing from previous run")
flags.DEFINE_boolean("time_training", False, "Print how long each step takes, instead of every 100 steps")
flags.DEFINE_boolean("moving_average", False, "Whether to use an exponential moving average of the weights rather than the weights directly (requires tensorflow_addons)")
flags.DEFINE_boolean("share_most_weights", True, "Instead of regularizing weights in heterogeneous domain adaptation, share same-shape weights")
flags.DEFINE_integer("debugnum", 50, "Specify exact log/model/images number to use rather than incrementing from last. (Don't pass both this and --debug at the same time.)")

flags.mark_flag_as_required("method")
flags.mark_flag_as_required("dataset")
flags.mark_flag_as_required("sources")
flags.mark_flag_as_required("uid")


def get_directory_names():
    """ Figure out the log and model directory names """
    prefix = "ucihar"+"-"+"0"+"-"+"yndaws"
    # Use the number specified on the command line (higher precedence than --debug)
    attempt = "95"
    print("Debugging attempt:", attempt)
    prefix += "-"+str(attempt)
    model_dir = os.path.join('example-models', prefix)
    log_dir = os.path.join('example-logs', prefix)
    return model_dir, log_dir


def get_models_to_evaluate():
    """
    Returns the models to evaluate based on what is in logdir and modeldir
    specified as command line arguments. The matching pattern is specified by
    the match argument.

    Returns: [(log_dir, model_dir, config), ...]
    """
files = pathlib.Path('example-logs').glob('ucihar-0-yndaws-11')
models_to_evaluate = []

for log_dir in files:
    config = file_utils.get_config(log_dir)
    model_dir = os.path.join('example-models', log_dir.stem)
    assert os.path.exists(model_dir), "Model does not exist "+str(model_dir)
    models_to_evaluate.append((str(log_dir), model_dir, config))

    return models_to_evaluate


def main(argv):
    # Allow running multiple at once
    set_gpu_memory(FLAGS.gpumem)
    # Figure out the log and model directory filenames
    assert FLAGS.uid != "", "uid cannot be an empty string"
    model_dir, log_dir = get_directory_names()

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Write config file about what dataset we're using, sources, target, etc.
    file_utils.write_config_from_args(log_dir)

    # Load datasets
    source_datasets, target_dataset = load_datasets.load_da(FLAGS.dataset,
        FLAGS.sources, FLAGS.target, test=FLAGS.test)
    # for x in source_datasets:
    #     print (x)
    # source_train_iterators = [iter(x.train) for x in source_datasets]
    # print (len(source_train_iterators))
    # for x in source_train_iterators:
    #     a = next(x)
    #     print (a)
    # data_sources = [next(x) for x in source_train_iterators]
    # data_sources = [next(x) for x in source_train_iterators]
    # data_sources = [next(x) for x in source_train_iterators]

    # Need to know which iteration for learning rate schedule
    global_step = tf.Variable(0, name="global_step", trainable=False)

    # Load the method, model, etc.
    method = methods.get_method(FLAGS.method,
        source_datasets=source_datasets,
        target_dataset=target_dataset,
        model_name=FLAGS.model,
        global_step=global_step,
        total_steps=FLAGS.steps,
        ensemble_size=FLAGS.ensemble,
        moving_average=FLAGS.moving_average,
        share_most_weights=FLAGS.share_most_weights)

    # Check that this method is supposed to be trainable. If not, we're done.
    # (Basically, we just wanted to write the config file for non-trainable
    # models.)


    if not method.trainable:
        print("Method not trainable. Exiting now.")
        return

    # Checkpoints
    checkpoint = tf.train.Checkpoint(
        global_step=global_step, **method.checkpoint_variables)
    checkpoint_manager = CheckpointManager(checkpoint, model_dir, log_dir)
    checkpoint_manager.restore_latest()

    # Metrics
    has_target_domain = target_dataset is not None
    metrics = Metrics(log_dir, method, source_datasets, target_dataset,
        has_target_domain)

    # Start training
    #
    # TODO maybe eventually rewrite this in the more-standard Keras way
    # See: https://www.tensorflow.org/guide/keras/train_and_evaluate
    for i in range(int(global_step), FLAGS.steps+1):
        t = time.time()
        data_sources, data_target = method.train_step()
        global_step.assign_add(1)
        t = time.time() - t

        if FLAGS.time_training:
            print(int(global_step), t, sep=",")
            continue  # skip evaluation, checkpointing, etc. when timing

        if i%1000 == 0:
            print("step %d took %f seconds"%(int(global_step), t))
            sys.stdout.flush()  # otherwise waits till the end to flush on Kamiak

        # Metrics on training/validation data
        if FLAGS.log_train_steps != 0 and i%FLAGS.log_train_steps == 0:
            metrics.train(data_sources, data_target, global_step, t)

        # Evaluate every log_val_steps but also at the last step
        validation_accuracy_source = None
        validation_accuracy_target = None
        if (FLAGS.log_val_steps != 0 and i%FLAGS.log_val_steps == 0) \
                or i == FLAGS.steps:
            validation_accuracy_source, validation_accuracy_target \
                = metrics.test(global_step)
            print(validation_accuracy_source,validation_accuracy_target)

        # Checkpoints -- Save either if at the right model step or if we found
        # a new validation accuracy. If this is better than the previous best
        # model, we need to make a new checkpoint so we can restore from this
        # step with the best accuracy.


        if (FLAGS.model_steps != 0 and i%FLAGS.model_steps == 0) \
                or validation_accuracy_source is not None:
            checkpoint_manager.save(int(global_step-1),
                validation_accuracy_source, validation_accuracy_target)

        # Plots
        if FLAGS.log_plots_steps != 0 and i%FLAGS.log_plots_steps == 0:
            metrics.plots(global_step)

    # We're done -- used for hyperparameter tuning
    file_utils.write_finished(log_dir)


if __name__ == "__main__":
    app.run(main)

from __future__ import print_function
import time
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
%matplotlib inline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


x=np.load('./matuwave.npz')
task_inv_r=x['task_inv_r']
domain_inv_r=x['domain_inv_r']
task_dep_r=x['task_dep_r']
domain_dep_r=x['domain_dep_r']
task_y_true=x['task_y_true']
domain_y_true=x['domain_y_true']

tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=300)
tsne_results = tsne.fit_transform(task_inv_r)

import pandas as pd

df_subset = pd.DataFrame(data=task_y_true,columns=['y'])
df_subset['tsne-2d-one'] = tsne_results[:,0]
df_subset['tsne-2d-two'] = tsne_results[:,1]
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    # palette=sns.color_palette("hls", 6),
    data=df_subset,
    legend="full",
    alpha=0.3
)
plt.show()



tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=300)
tsne_results = tsne.fit_transform(task_dep_r_2)

import pandas as pd

df_subset = pd.DataFrame(data=task_y_true_2,columns=['y'])
df_subset['tsne-2d-one'] = tsne_results[:,0]
df_subset['tsne-2d-two'] = tsne_results[:,1]
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("hls", 6),
    data=df_subset,
    legend="full",
    alpha=0.3
)
plt.show()

filename='matuwave'

x=np.load('./'+filename+'.npz')
task_inv_r=x['task_inv_r']
domain_inv_r=x['domain_inv_r']
task_dep_r=x['task_dep_r']
domain_dep_r=x['domain_dep_r']
task_y_true=x['task_y_true']
domain_y_true=x['domain_y_true']

y=np.load('./'+filename+'0.npz')
task_inv_r_2=y['task_inv_r']
domain_inv_r_2=y['domain_inv_r']
task_dep_r_2=y['task_dep_r']
domain_dep_r_2=y['domain_dep_r']
task_y_true_2=y['task_y_true']
domain_y_true_2=y['domain_y_true']


# import pandas as pd
#
# df_subset = pd.DataFrame(data=np.concatenate((domain_y_true,domain_y_true_2),axis=0),columns=['y'])
# df_subset['tsne-2d-one'] = np.concatenate((domain_inv_r,domain_inv_r_2),axis=0)[:,0]
# df_subset['tsne-2d-two'] = np.concatenate((domain_inv_r,domain_inv_r_2),axis=0)[:,1]
# plt.figure(figsize=(16,10))
# # plt.rcParams["axes.labelsize"] = 30
# sns.scatterplot(
#     x="tsne-2d-one", y="tsne-2d-two",
#     hue="y",
#     palette=sns.color_palette("hls", 2),
#     data=df_subset,
#     legend="full",
#     alpha=0.3
# )
# plt.show()
# plt.figure(figsize=(16,10))

df_subset = pd.DataFrame(data=np.concatenate((domain_y_true.astype(int),domain_y_true_2.astype(int)),axis=0),columns=['Domain'])
df_subset['tsne-x'] = np.concatenate((domain_dep_r,domain_dep_r_2),axis=0)[:,0]
df_subset['tsne-y'] = np.concatenate((domain_dep_r,domain_dep_r_2),axis=0)[:,1]
sns.set(font_scale=2)
sns.set_style("white")


sns_plot=sns.scatterplot(
    x="tsne-x", y="tsne-y",
    hue="Domain",
    palette=sns.color_palette("hls", 2),
    data=df_subset,
    legend="full",
    alpha=0.3
)
sns_plot.set_ylabel('')
sns_plot.set_xlabel('')
sns_plot.set(yticks=[])
sns_plot.set(xticks=[])
sns_plot.despine(top=True, right=True, left=True, bottom=True)
sns_plot.get_legend().remove()
sns_plot.figure.savefig("output"+filename+".png")
plt.show()



df_subset = pd.DataFrame(data=np.concatenate((domain_y_true.astype(int),domain_y_true_2.astype(int)),axis=0),columns=['Domain'])
df_subset['tsne-x'] = np.concatenate((domain_inv_r,domain_inv_r_2),axis=0)[:,0]
df_subset['tsne-y'] = np.concatenate((domain_inv_r,domain_inv_r_2),axis=0)[:,1]
sns.set(font_scale=2)
sns.set_style("white")


sns_plot=sns.scatterplot(
    x="tsne-x", y="tsne-y",
    hue="Domain",
    palette=sns.color_palette("hls", 2),
    data=df_subset,
    legend="full",
    alpha=0.3
)
sns_plot.set_ylabel('')
sns_plot.set_xlabel('')
sns_plot.set(yticks=[])
sns_plot.set(xticks=[])
sns_plot.get_legend().remove()
sns_plot.figure.savefig("output"+filename+"2.png")
plt.show()



tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=300)
tsne_results = tsne.fit_transform(task_inv_r)


df_subset = pd.DataFrame(data=task_y_true,columns=['Task'])
df_subset['tsne-x'] = tsne_results[:,0]
df_subset['tsne-y'] = tsne_results[:,1]
sns.set(font_scale=2)
sns.set_style("white")


sns_plot=sns.scatterplot(
    x="tsne-x", y="tsne-y",
    hue="Task",
    palette=sns.color_palette("hls"),
    data=df_subset,
    legend="full",
    alpha=0.3
)
sns_plot.set_ylabel('')
sns_plot.set_xlabel('')
sns_plot.set(yticks=[])
sns_plot.set(xticks=[])
sns_plot.get_legend().remove()
sns_plot.despine(top=True, right=True, left=True, bottom=True)
sns_plot.set_linewidth(0.0)
sns_plot.figure.savefig("output"+filename+"3.png")
plt.show()

tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=300)
tsne_results = tsne.fit_transform(task_dep_r)

import matplotlib as mpl
mpl.rcParams['font.family'] = 'Times'

df_subset = pd.DataFrame(data=task_y_true.astype(int),columns=['Task'])
df_subset['tsne-x'] = tsne_results[:,0]
df_subset['tsne-y'] = tsne_results[:,1]
sns.set(font_scale=2)
sns.set_style("white")


sns_plot=sns.scatterplot(
    x="tsne-x", y="tsne-y",
    hue="Task",
    palette=sns.color_palette("hls",8),
    data=df_subset,
    legend="full",
    alpha=0.3
)
sns_plot.set_ylabel('')
sns_plot.set_xlabel('')
sns_plot.set(yticks=[])
sns_plot.set(xticks=[])
sns_plot.despine(top=True, right=True, left=True, bottom=True)
sns_plot.set_linewidth(0.0)
sns_plot.figure.savefig("output"+filename+"4.png")
plt.show()
