# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# changes made by Johan Weiss (s182969) to fit the Terrorist attack dataset.
# the original script can be found here: https://raw.githubusercontent.com/tensorflow/neural-structured-learning/master/neural_structured_learning/examples/preprocess/cora/preprocess_cora_dataset.py


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import random
import time

from absl import app
from absl import flags
from absl import logging
from neural_structured_learning.tools import graph_utils
import six
import tensorflow as tf

FLAGS = flags.FLAGS
FLAGS.showprefixforinfo = False

flags.DEFINE_string(
    'input_content', '/tmp/terrorist_attack.nodes',
    """Input file for terror attack content that contains ID, features and labels.""")
flags.DEFINE_string('input_graph', '/tmp/terrorist_attack_loc_org.edges',
                    """Input file for terror attack citation graph in TSV format.""")
flags.DEFINE_integer(
    'max_nbrs', None,
    'The maximum number of neighbors to merge into each labeled Example.')
flags.DEFINE_float(
    'train_percentage', None,
    """The percentage of examples to be created as training data. The rest
    are created as test data.""")
flags.DEFINE_string(
    'output_train_data', '/tmp/train_merged_examples.tfr',
    """Output file for training data merged with graph in TF Record format.""")
flags.DEFINE_string('output_test_data', '/tmp/test_examples.tfr',
                    """Output file for test data in TF Record format.""")
flags.DEFINE_string('output_valid_data', '/tmp/valid_examples.tfr',
                    """Output file for test data in TF Record format.""")


def _int64_feature(*value):
  """Returns int64 tf.train.Feature from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=list(value)))


def parse_content(in_file, train_percentage):

  label_index = {
      'Arson': 0,
      'Bombing': 1,
      'Kidnapping': 2,
      'NBCR_Attack': 3,
      'other_attack': 4,
      'Weapon_Attack': 5,
  }
  # Fixes the random seed so the train/test split can be reproduced.
  random.seed(1)
  train_examples = {}
  test_examples = {}
  valid_examples = {}
  with open(in_file, 'rU') as terror_content:
    for line in terror_content:
      entries = line.rstrip('\n').split('\t')
      # entries contains [ID, Word1, Word2, ..., Label]; 'Words' are 0/1 values.
      attributes = map(int, entries[1:-1])
      # print("feature", *words)
      features = {
          'attributes': _int64_feature(*attributes),
          'label': _int64_feature(label_index[entries[-1]]),

      }
      example_features = tf.train.Example(
          features=tf.train.Features(feature=features))

      example_id = entries[0]
      # print("example id", example_id)
      if random.uniform(0, 1) <= train_percentage:  # for train/test split.
        train_examples[example_id] = example_features
      else:
        if random.uniform(0, 1) <= 0.5:
          test_examples[example_id] = example_features
        else:
          valid_examples[example_id] = example_features

  return train_examples, test_examples, valid_examples


def _join_examples(seed_exs, nbr_exs, graph, max_nbrs):

  out_degree_count = collections.Counter()

  def has_ex(node_id):
    # print("Node id:", node_id, '/n')
    """Returns true iff 'node_id' is in the 'seed_exs' or 'nbr_exs dict'."""
    result = (node_id in seed_exs) or (node_id in nbr_exs)
    # ignore the warning (we know it's there)- it makes the traning code look messy
    # if not result:
    #   logging.warning('No tf.train.Example found for edge target ID: "%s"',
    #                   node_id)
    return result

  def lookup_ex(node_id):
    """Returns the Example from `seed_exs` or `nbr_exs` with the given ID."""
    return seed_exs[node_id] if node_id in seed_exs else nbr_exs[node_id]

  def join_seed_to_nbrs(seed_id):

    nbr_dict = graph[seed_id] if seed_id in graph else {}
    nbr_wt_ex_list = [(nbr_wt, nbr_id)
                      for (nbr_id, nbr_wt) in six.iteritems(nbr_dict)
                      if has_ex(nbr_id)]
    result = sorted(nbr_wt_ex_list, reverse=True)[:max_nbrs]
    out_degree_count[len(result)] += 1
    return result

  def merge_examples(seed_ex, nbr_wt_ex_list):
    #Merges neighbor Examples into the given seed Example `seed_ex`.

    # Make a deep copy of the seed Example to augment.
    merged_ex = tf.train.Example()
    merged_ex.CopyFrom(seed_ex)

    # Add a feature for the number of neighbors.
    merged_ex.features.feature['NL_num_nbrs'].int64_list.value.append(
        len(nbr_wt_ex_list))

    # Enumerate the neighbors, and merge in the features of each.
    for index, (nbr_wt, nbr_id) in enumerate(nbr_wt_ex_list):
      prefix = 'NL_nbr_{}_'.format(index)
      # Add the edge weight value as a new singleton float feature.
      weight_feature = prefix + 'weight'
      merged_ex.features.feature[weight_feature].float_list.value.append(nbr_wt)
      # Copy each of the neighbor Examples features, prefixed with 'prefix'.
      nbr_ex = lookup_ex(nbr_id)
      for (feature_name, feature_val) in six.iteritems(nbr_ex.features.feature):
        new_feature = merged_ex.features.feature[prefix + feature_name]
        new_feature.CopyFrom(feature_val)
    return merged_ex

  start_time = time.time()
  logging.info(
      'Joining seed and neighbor tf.train.Examples with graph edges...')
  for (seed_id, seed_ex) in six.iteritems(seed_exs):
    yield merge_examples(seed_ex, join_seed_to_nbrs(seed_id))
  logging.info(
      'Done creating and writing %d merged tf.train.Examples (%.2f seconds).',
      len(seed_exs), (time.time() - start_time))
  logging.info('Out-degree histogram: %s', sorted(out_degree_count.items()))


def main(unused_argv):
  start_time = time.time()

  # Parses terrorist attack content into TF Examples.
  train_examples, test_examples, valid_examples = parse_content(FLAGS.input_content,
                                                     FLAGS.train_percentage)

  graph = graph_utils.read_tsv_graph(FLAGS.input_graph)
  # print(graph)
  graph_utils.add_undirected_edges(graph)

  # Joins 'train_examples' with 'graph'. 'test_examples' are used as *unlabeled*
  # neighbors for transductive learning purpose. In other words, the labels of
  # test_examples are not used.
  with tf.io.TFRecordWriter(FLAGS.output_train_data) as writer:
    for merged_example in _join_examples(train_examples, test_examples, graph,
                                         FLAGS.max_nbrs):
      writer.write(merged_example.SerializeToString())

  logging.info('Output training data written to TFRecord file: %s.',
               FLAGS.output_train_data)

  # Writes 'test_examples' out w/o joining with the graph since graph
  # regularization is used only during training, not testing/serving.
  with tf.io.TFRecordWriter(FLAGS.output_test_data) as writer:
    for example in six.itervalues(test_examples):
      writer.write(example.SerializeToString())

  with tf.io.TFRecordWriter(FLAGS.output_valid_data) as writer:
    for example in six.itervalues(valid_examples):
      writer.write(example.SerializeToString())

  logging.info('Output test data written to TFRecord file: %s.',
               FLAGS.output_test_data)
  logging.info('Output valid data written to TFRecord file: %s.',
               FLAGS.output_valid_data)
  logging.info('Total running time: %.2f minutes.',
               (time.time() - start_time) / 60.0)


if __name__ == '__main__':
  # Ensures TF 2.0 behavior even if TF 1.X is installed.
  tf.compat.v1.enable_v2_behavior()
  app.run(main)
