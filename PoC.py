#!/usr/bin/env python3

import cv2
import numpy as np
import pickle

import tensorflow as tf


imsize = 32

def sliceImage(img):
  tile_num = (8, 34 * 5)

  sy = img.shape[0] / tile_num[0]
  sx = img.shape[1] / tile_num[1]

  print("Input tile size:", sx, sy)

  nimg = cv2.resize(img, (imsize*tile_num[1], imsize*tile_num[0]))

  rows = [[] for x in range(tile_num[0])]
  for i in range(tile_num[1]):
    for j in range(tile_num[0]):
      tile = nimg[imsize*j:imsize*(j+1), imsize*i:imsize*(i+1)]
      rows[j].append(tile)

  tiles = np.array(rows)
  return tiles

def augment(img, num):
  result = []

  for i in range(num):
    angle = (np.random.random() - 0.5) * 40
    scale = 0.65 + np.random.random() * 0.35

    mat = cv2.getRotationMatrix2D((imsize/2, imsize/2), angle, scale)

    # shift
    mat[0, 2] += (np.random.random() - 0.5) * imsize * 0.2
    mat[1, 2] += (np.random.random() - 0.5) * imsize * 0.2

    dst = cv2.warpAffine(img, mat, (imsize, imsize), borderValue = (0.5, 0.5, 0.5))  
    result.append(dst)
    # cv2.imshow("kek", dst)
    # cv2.waitKey()

  return result
  
def annotateTiles(tiles):
  xs = []
  labels = []
  for ri, row in enumerate(tiles):
    for ci, tile in enumerate(row):
      if ci < (34*4):
        xs.append(tile)
        labels.append(ri+1)
      elif ri < 4:
        xs.append(tile)
        labels.append(0)

  assert(len(xs) == len(labels))

  return (np.array(xs), np.array(labels))

def loadDataset(cached = True):
  if not cached:
    img = cv2.imread("Variant2fixed4783.png", cv2.IMREAD_GRAYSCALE)
    print("Input:", img.shape)

    tiles = sliceImage(img)
    print("Tiled array:", tiles.shape)

    features, labels = annotateTiles(tiles)

    # shuffle data
    permutation = np.random.permutation(range(labels.size))
    features = [features[i] for i in permutation]
    labels = [labels[i] for i in permutation]

    # split on train, devel, test
    num = len(labels)
    splits = list(map(round, [num * 0.8, num * 1.0, num])) # 0.8, 0.15, 0.05


    dataset = dict(
      train = (features[0:splits[0]], labels[0:splits[0]]),
      devel = (features[splits[0]:splits[1]], labels[splits[0]:splits[1]]),
      test = (features[splits[1]:splits[2]], labels[splits[1]:splits[2]]),
    )

    # augment training dataset
    train_features = []
    train_labels = []

    augmentation_rate = 200
    for i in range(splits[0]):
      train_features += augment(dataset["train"][0][i], augmentation_rate)
      train_labels += [dataset["train"][1][i]] * augmentation_rate

    # reshuffle training data
    permutation = np.random.permutation(range(len(train_labels)))
    train_features = [train_features[i] for i in permutation]
    train_labels = [train_labels[i] for i in permutation]
    print(len(train_labels))

    dataset["train"] = (train_features, train_labels)

    with open("dataset.pickle", "wb") as f:
      pickle.dump(dataset, f)

  with open("dataset.pickle", "rb") as f:
    dataset = pickle.load(f)

  return dataset

def cnnModelFn(features, labels, mode):
  input_layer = tf.reshape(features["x"], [-1, imsize, imsize, 1])

  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=24,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=32,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  pool2_flat = tf.reshape(pool2, [-1, (imsize // 4) ** 2 * 32])
  dense1 = tf.layers.dense(inputs=pool2_flat, units=256, activation=tf.nn.relu)
  dropout1 = tf.layers.dropout(inputs=dense1, rate=0.1, training=(mode == tf.estimator.ModeKeys.TRAIN))

  logits = tf.layers.dense(inputs=dropout1, units=9)

  predictions = {
      "classes": tf.argmax(input=logits, axis=1),
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  if mode == tf.estimator.ModeKeys.TRAIN:
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.005)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.00005)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(unused_argv):
  dataset = loadDataset(True)

  train_data = np.asarray(dataset["train"][0], dtype=np.float32) / 255
  train_labels = np.asarray(dataset["train"][1], dtype=np.int32)
  eval_data = np.asarray(dataset["devel"][0], dtype=np.float32) / 255
  eval_labels = np.asarray(dataset["devel"][1], dtype=np.int32)

  #for img in train_data:
  #  print(img)
  #  cv2.imshow("kek", img)
  #  cv2.waitKey()
  #return

  roman_classifier = tf.estimator.Estimator(
      model_fn=cnnModelFn, model_dir="/tmp/roman_convnet_model")

  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

  # Feed functions
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=100,
      num_epochs=None,
      shuffle=True)

  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)

  with tf.Session() as sess:
    # Train the model
    for i in range(40):
      roman_classifier.train(
        input_fn=train_input_fn,
        steps=500,
        hooks=[logging_hook])
      eval_results = roman_classifier.evaluate(input_fn=eval_input_fn)
      print(eval_results)

    predicted_ids = []
    predictions = roman_classifier.predict(input_fn=eval_input_fn)
    print(predictions)
    for pred_dict, expect, i in zip(predictions, eval_labels, range(eval_labels.size)):
      class_id = pred_dict['classes']
      predicted_ids.append(class_id)

      if class_id != expect:
        print(str(class_id), " true: ", expect)
        cv2.imshow("Err", eval_data[i])
        cv2.waitKey()

    conf_matrix_op = tf.confusion_matrix(eval_labels, predicted_ids, num_classes = 9)
    conf_matrix = sess.run(conf_matrix_op)
    print(conf_matrix)

    eval_results = roman_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


if __name__ == '__main__':
  tf.app.run()
