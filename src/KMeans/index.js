// Copyright (c) 2019 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/* eslint "no-param-reassign": [2, { "props": false }] */
/*
K-Means Algorithm (with Euclidean distance).
*/

import * as tf from '@tensorflow/tfjs';
import callCallback from '../utils/callcallback';
import {randomSample} from '../utils/random';

/**
 * @typedef {Object} KMeansOptions
 * @property {number} k - The number of clusters. Default 3.
 * @property {number} maxIter - Max number of iterations to try before forcing convergence. Default 5.
 * @property {number} threshold - Threshold for updated centroid distance before declaring convergence. Default 0.5
 */

/**
 * @type {KMeansOptions}
 */
const DEFAULTS = {
  'k': 3,
  'maxIter': 5,
  'threshold': 0.5,
};

// TODO: can maybe combine with csv read in NeuralNetwork
/**
 * Read in a csv file from a path to its location.
 * @param {string} path
 * @return {Promise<tf.TensorContainer[]>}
 */
async function readCsv(path) {
  const myCsv = tf.data.csv(path);
  return myCsv.toArray();
}

/**
 * A valid dataset is an array of x,y coordinates.  Each element of the array should be an object with numeric
 * properties like `{ x: 0, y: 0 }`, or an array of numbers like `[0, 0]`.
 * @typedef {Array<(Record<any, number> | Array<number>)>} KMeansDataSet
 */

/**
 * Load and flatten an array of arrays, an array of objects, or a string
 *   path to a csv.
 * @param {(string|KMeansDataSet)} inputData
 * @return {number[][]}
 */
async function loadDataset(inputData) {
  let data;
  if (typeof inputData === 'string') {
    data = await readCsv(inputData);
  } else {
    data = inputData;
  }
  return data.map((d) => {
    return Object.values(d)
  });
}

/**
 * // listed on https://learn.ml5js.org/
 * @property {KMeansOptions} config
 * @property {Array<[number, number] & {tensor: tf.Tensor<tf.Rank.R1>, centroid: number}>} dataset
 * @property {tf.Tensor} dataTensor
 * @property {tf.Tensor2D} centroids
 *
 * // not listed
 * @property {Promise} ready
 * @property {tf.Tensor2D} centroidsOld
 */
class KMeans {
  /**
   * Create a K-Means.
   * @param {(string | KMeansDataSet)} dataset - The dataset to cluster.
   * @param {Partial<KMeansOptions>} [options] - An object describing a model's parameters:
   *    - k: number of clusters
   *    - maxIter: Max number of iterations to try before forcing convergence.
   *    - threshold: Threshold for updated centroid distance before declaring convergence.
   * @param {function} [callback] - Optional. A callback to be called once
   *    the model has loaded. If no callback is provided, it will return a 
   *    promise that will be resolved once the model has loaded.
   */
  constructor(dataset, options = {}, callback) {
    this.config = {
      ...options,
      ...DEFAULTS
    };
    this.ready = callCallback(this.load(dataset), callback);
  }

  /**
   * Load dataset, find initial centroids, and run model.
   * Returns a Promise which resolves to the model instance when ready
   *
   * @public
   * @param {(string | KMeansDataSet)} dataset
   * @return {Promise<this>}
   */
  async load(dataset) {
    this.dataset = await loadDataset(dataset);
    tf.tidy( () => {
      this.dataTensor = tf.tensor2d(this.dataset);
      this.dataset.forEach(d => {
        // note: Object.values(d) will not work if d.tensor and/or d.centroid have already been set
        // but it is ok here because this.dataset was just set
        d.tensor = tf.tensor1d(Object.values(d));
      });
      this.centroids = tf.tensor2d(randomSample(this.dataset, this.config.k, false));
      this.fit();
    })
    return this;
  }

  /**
   * Run K-Means algorithm.
   * @private
   */
  fit() {
    this.getClosestCentroids()
    this.recenterCentroids();
    let centroidDistance = KMeans.getEuclideanDistance(this.centroids, this.centroidsOld);
    let iteration = 0;
    while (centroidDistance > this.config.threshold && iteration < this.config.maxIter) {
      this.getClosestCentroids();
      this.recenterCentroids();
      centroidDistance = KMeans.getEuclideanDistance(this.centroids, this.centroidsOld);
      iteration += 1;
    }
  }

  /**
   * Find closest centroids to each observation and store as an attribute.
   * @private
   */
  getClosestCentroids() {
    // find closest initial tensor
    this.dataset.forEach(d => {
      d.centroid = this.closestCentroid(d.tensor);
    })
  }

  /**
   * Find closest centroid for a given tensor
   * @private
   * @param {tf.Tensor} dataTensor
   * @return {number}
   */
  closestCentroid(dataTensor) {
    return tf.tidy(() => {
      const dist = this.centroids.squaredDifference(dataTensor).sum(1).sqrt();
      return dist.argMin().arraySync();
    });
  }

  /**
   * Assign `value` to a cluster.
   * @private
   * Note: Cannot make public because of Error when calling: "Tensor is disposed" on this.centroids.  Would need work.
   *
   * @param {(number[] | Record<any, number>)} value - a single point object `{ x: 0, y: 0 }` or tuple `[0, 0]`
   * @return
   */
  classify(value) {
    return tf.tidy(() => {
      // input must be array or object
      const valueTensor = tf.tensor1d(Object.values(value));
      return this.closestCentroid(valueTensor);
    })
  }

  /**
   * Recenter each centroid.
   * @private
   */
  recenterCentroids() {
    // store previous run's centroids for convergence
    // recenter each centroid
    this.centroids = tf.tidy(() => {
      this.centroidsOld = this.centroids;
      return tf.stack(this.centroids.unstack().map((centroid, k) => {
        // subset centroid to its cluster
        const centroidK = this.dataset.filter(d => d.centroid === k);
        // convert to tensor
        const centroidKTensor = centroidK.map(d => d.tensor);
        if (centroidKTensor.length === 0) {
          return centroid;
        } else if (centroidKTensor.length === 1) {
          return centroidKTensor[0];
        }
        // grab mean for for cluster
        // TODO: does it make sense to have one tidy inside another?
        return tf.tidy(() => tf.stack(centroidKTensor).mean(0))
      }))
    });
  }

  /**
   * Calculate the Euclidean distance between two tensors.
   * @param {tf.Tensor2D} tensor1
   * @param {tf.Tensor2D} tensor2
   * @return {number}
   */
  static getEuclideanDistance(tensor1, tensor2) {
    // calculate euclidean distance between two arrays
    const distTensor = tf.tidy(() => {
      const distance = tf.squaredDifference(tensor1, tensor2).sum().sqrt();
      return distance.dataSync()
    })
    return distTensor[0];
  }
}

const kmeans = (dataset, options, callback) => new KMeans(dataset, options, callback);

export default kmeans;