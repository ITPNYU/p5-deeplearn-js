// Copyright (c) 2018 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT


/**
 * @typedef {Object} Classification
 * @param {string} className - The name of the matched class.
 * @param {number} probability - The probability from 0 to 1 that this class name is a match.
 */

/**
 * Utility function which limits classifier predictions to a provided number `topK`
 *
 * @param {number[] | Float32Array} values - Array of probabilities
 * where the indexes match the indexes of the provided classes
 * and the values of the probability that that class is a match.
 * @param {number} topK - Number of top classes to return.
 * @param {string[]} CLASSES - Array of labels.
 * @return {Classification[]} - Array of objects with properties probability and className.
 */
export function getTopKClassesFromArray(values, topK, CLASSES) {
  const labeled = [...values].map((value, i) => ({
    className: CLASSES[i],
    probability: value
  }));
  // note: the performance would be better if we don't sort the whole array
  // we only need the top k, so the order of others doesn't matter.
  labeled.sort((a, b) => b.probability - a.probability);
  return labeled.slice(0, topK);
}

/**
 * @param {Tensor} logits - Tensorflow Tensor, likely of type "float32"
 * @param {number} topK - Number of top classes to return.
 * @param {string[]} CLASSES - Array of labels.
 * @return {Promise<Classification[]>} - Array of objects with properties probability and className.
 */
export async function getTopKClassesFromTensor(logits, topK, CLASSES) {
  const values = await logits.data();
  return getTopKClassesFromArray(values, topK, CLASSES);
}

export default { getTopKClassesFromArray, getTopKClassesFromTensor };
