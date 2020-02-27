// Copyright (c) 2018 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/**
 * a small utility check to see if `toBeDetermined` is a `tf.Tensor` or a `tf.Tensor[]`
 * @param  toBeDetermined `tf.Tensor` || `tf.Tensor[]`
 * @returns a `boolean` indicating if it's a `tf.Tensor` or a `tf.Tensor[]`
 */
export default function isTensorArray(toBeDetermined) {
	return !toBeDetermined.shape;
}
