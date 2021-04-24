// Copyright (c) 2018 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/*
General Feature Extractor Manager
*/

import Mobilenet from './Mobilenet';

/**
 * @typedef {Object} FeatureExtractorOptions
 * @property {number} version - default 1
 * @property {number} alpha - default 1.0
 * @property {number} topk - default 3
 * @property {number} learningRate - default 0.0001
 * @property {number} hiddenUnits - default 100
 * @property {number} epochs - default 20
 * @property {number} numClasses - default 2
 * @property {number} batchSize - default 0.4
 */


/**
 * Create a featureExtractor.
 * @param {string} model - The model from which extract the learned features. Case-insensitive
 * @param {(FeatureExtractorOptions | function)} [optionsOrCallback] - Optional.
 * @param {function} [cb] - Optional.
 */
const featureExtractor = (model, optionsOrCallback, cb) => {
  let modelName;
  if (typeof model !== 'string') {
    throw new Error('Please specify a model to use. E.g: "MobileNet"');
  } else {
    modelName = model.toLowerCase();
  }

  let options = {};
  let callback = cb;

  if (typeof optionsOrCallback === 'object') {
    options = optionsOrCallback;
  } else if (typeof optionsOrCallback === 'function') {
    callback = optionsOrCallback;
  }

  if (modelName === 'mobilenet') {
    return new Mobilenet(options, callback);
  }
  throw new Error(`${modelName} is not a valid model.`);
};

export default featureExtractor;
