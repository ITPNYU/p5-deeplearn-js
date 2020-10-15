// Copyright (c) 2020 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/* eslint "no-param-reassign": [2, { "props": false }] */
/*
DBSCAN Algorithm (with Euclidian distance). Influenced By jDBSCAN
*/

import * as tf from "@tensorflow/tfjs";
import callCallback from "../utils/callcallback";

/**
 * Read in a csv file from a path to its location.
 * @param {string} path
 */
async function readCsv(path) {
  const myCsv = tf.data.csv(path);
  const loadedData = await myCsv.toArray();
  return loadedData;
}

/**
 * Load and flatten an array of arrays, an array of objects, or a string
 *   path to a csv.
 * @param {string || array || object} inputData
 */
async function loadDataset(inputData) {
  let data;
  if (typeof inputData === "string") {
    data = await readCsv(inputData);
  } else {
    data = inputData;
  }
  const dataFlat = data.map(d => {
    return Object.values(d);
  });
  return dataFlat;
}

const DEFAULTS = {
  eps: 50,
  minPts: 3,
};

class DBSCAN {
  /**
   * Create a DBSCAN.
   * @param {String || array || object} dataset - The dataset to cluster. in x, y format => [{x:1,y:2}]
   * @param {options} options - An object describing a model's parameters:
   *    - eps: Minimum distance between neighbours
   *    - minPts: Minimum number of neighbours to count as a core point
   * @param {function} callback  - Optional. A callback to be called once
   *    the model has loaded. If no callback is provided, it will return a
   *    promise that will be resolved once the model has loaded.
   */

  constructor(dataset, options, callback) {
    this.config = {
      eps: options.eps || DEFAULTS.eps,
      minPts: options.minPts || DEFAULTS.minPts,
    };
    this.lastClusterId = 0;
    this.status = [];
    this.ready = callCallback(this.load(dataset), callback);
  }

  /**
   * Load dataset, and run model.
   * @param {string || array || object} dataset
   */
  async load(dataset) {
    this.dataset = await loadDataset(dataset);
    tf.tidy(() => {
      this.dataTensor = tf.tensor2d(this.dataset);
      this.dataset.forEach(d => {
        const tensors = tf.tensor1d(Object.values(d));
        d.tensor = tensors;
      });
      this.fit();
    });
    return this;
  }

  /**
   * Run DBSCAN algorithm.
   */
  fit() {
    this.dataset.forEach((d, idx) => {
      if (d.status === undefined) {
        d.status = 0; // initlize as a noise point
        const neighboursIndices = this.getNeighboursIndices(d);
        if (neighboursIndices.length < this.config.minPts) {
          // Border or noise
          d.status = 0;
        } else {
          this.incrementClusterId();
          this.extend(idx, neighboursIndices);
        }
      }
    });
  }

  /**
   * Extend cluster by running algorithm on neighbours and detect neighbours that are core points as well
   * @param {number} pointIndex
   * @param {number[]} neighboursIndices
   */
  extend(pointIndex, neighboursIndices) {
    this.dataset[pointIndex].clusterid = this.getClusterId();
    this.dataset[pointIndex].status = this.dataset[pointIndex].clusterid;
    neighboursIndices.forEach(neighbourIndex => {
      if (this.dataset[neighbourIndex].status === undefined) {
        // Status unknown intialize as noise
        this.dataset[neighbourIndex].status = 0;
        const currNeighbours = this.getNeighboursIndices(
          // Neighbours of this point
          this.dataset[neighbourIndex],
        );
        const currNumNeighbours = currNeighbours.length;

        if (currNumNeighbours >= this.config.minPts) {
          // If Neighbours are above minimum we go further and add this and potential neighbours to clusterId
          this.extend(neighbourIndex, currNeighbours);
        }
      }
      if (this.dataset[neighbourIndex].status < 1) {
        this.dataset[neighbourIndex].status = this.dataset[pointIndex].clusterid;
        this.dataset[neighbourIndex].clusterid = this.dataset[pointIndex].clusterid;
      }
    });
  }

  /**
   * Return last generated cluster id
   */
  getClusterId() {
    return this.lastClusterId;
  }
  /**
   * increment cluster id
   */
  incrementClusterId() {
    this.lastClusterId += 1;
  }

  /**
   * Find closest neighbours to each observation.
   */
  getNeighboursIndices(point) {
    try {
      const neighbours = tf.tidy(() => {
        const { values, indices } = tf
          .squaredDifference(point.tensor, this.dataTensor)
          .sum(1)
          .sqrt()
          .topk(this.dataTensor.shape[0], true);
        return tf
          .stack([values.asType("float32"), indices.asType("float32")], 1)
          .arraySync()
          .filter(v => {
            return v[0] <= this.config.eps;
          })
          .reduce((prev, cur) => {
            prev.push(cur[1]);
            return prev;
          }, []);
      });
      return neighbours || [];
    } catch (error) {
      console.log(`error ${error}`);
    }
    return [];
  }
}

const dbscan = (dataset, options, callback) => new DBSCAN(dataset, options, callback);

export default dbscan;
