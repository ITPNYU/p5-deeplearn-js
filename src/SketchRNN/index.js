// Copyright (c) 2018 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/*
SketchRNN
*/

import * as ms from '@magenta/sketch';
import callCallback from '../utils/callcallback';
import modelPaths from './models';
import modelLoader from '../utils/modelLoader';

/**
 * Combine a model name like 'cat' and a size into a the absolute URL for the model on google cloud storage
 *
 * @param {string} model
 * @param {boolean} large
 * @return {string}
 */
const createPath = (model, large) => {
    return `https://storage.googleapis.com/quickdraw-models/sketchRNN/${large ? "large_models" : "models"}/${model}.gen.json`
}

/**
 * @typedef {Object} SketchRNNOptions
 * @property {number} temperature - default 0.65
 * @property {number} pixelFactor - default 3.0
 * @property {boolean} large - default true
 */
const DEFAULTS = {
    temperature: 0.65,
    pixelFactor: 3.0,
    large: true,
}

/**
 * @property {SketchRNNOptions} config
 * @property {ms.SketchRNN} model - magenta sketch model
 * @property {Promise} ready
 */
class SketchRNN {
    /**
     * Create SketchRNN.
     * @param {String} model - The name of the sketch model to be loaded.
     *    The names can be found in the models.js file
     * @param {Partial<SketchRNNOptions>} options
     * @param {function} callback - Optional. A callback function that is called once the model has loaded. If no callback is provided, it will return a promise
     *    that will be resolved once the model has loaded.
     */
    constructor(model, options, callback) {
        this.config = {
            ...DEFAULTS,
            ...options
        };
        // see if the model is an accepted name or a URL
        const modelUrl = modelPaths.has(model) ? createPath(model, this.config.large) : modelLoader.getModelPath(model);
        // create the model
        this.model = new ms.SketchRNN(modelUrl);
        this.penState = this.model.zeroInput();
        this.ready = callCallback(this.model.initialize(), callback);
    }

  /**
   * @param {Partial<SketchRNNOptions>} options
   * @param {number[][]} strokes
   * @return {Promise<{SketchRNNSeed}>}
   */
    async generateInternal(options, strokes) {
        const temperature = +options.temperature || this.config.temperature;
        const pixelFactor = +options.pixelFactor || this.config.pixelFactor;

        await this.ready;
        if (!this.rnnState) {
            this.rnnState = this.model.zeroState();
            this.model.setPixelFactor(pixelFactor);
        }

        if (Array.isArray(strokes) && strokes.length) {
            this.rnnState = this.model.updateStrokes(strokes, this.rnnState);
        }
        this.rnnState = this.model.update(this.penState, this.rnnState);
        const pdf = this.model.getPDF(this.rnnState, temperature);
        // note: this.penState is not used anywhere and doesn't need to be a property
        this.penState = this.model.sample(pdf);
        const result = {
            dx: this.penState[0],
            dy: this.penState[1],
        };
        if (this.penState[2] === 1) {
            result.pen = 'down';
        } else if (this.penState[3] === 1) {
            result.pen = 'up';
        } else if (this.penState[4] === 1) {
            result.pen = 'end';
        }
        return result;
    }

  /**
   * @typedef {Object} SketchRNNSeed
   * @property {number} dx - The x location of the pen.
   * @property {number} dy - The y location of the pen.
   * @property {string} pen - One of: 'up', 'down', 'end'.
   *    Determines whether the pen is down, up, or if the stroke has ended.
   *
   *
   * @param {SketchRNNSeed[] | Partial<SketchRNNOptions> | function} optionsOrSeedOrCallback
   * @param {Partial<SketchRNNOptions> | function} [optionsOrCallback]
   * @param {function} [cb]
   * @return {Promise<SketchRNNSeed>}
   */
    async generate(optionsOrSeedOrCallback, optionsOrCallback, cb) {
        let callback;
        let options;
        let seedStrokes;

        if (typeof optionsOrSeedOrCallback === 'function') {
            options = {};
            seedStrokes = [];
            callback = optionsOrSeedOrCallback;
        } else if (Array.isArray(optionsOrSeedOrCallback)) {
            options = {};
            seedStrokes = optionsOrSeedOrCallback;
            callback = optionsOrCallback;
        } else if (typeof optionsOrCallback === 'function') {
            options = optionsOrSeedOrCallback || {};
            seedStrokes = [];
            callback = optionsOrCallback;
        } else {
            options = optionsOrSeedOrCallback || {};
            seedStrokes = optionsOrCallback || [];
            callback = cb;
        }

        const strokes = seedStrokes.map(s => {
            const up = s.pen === 'up' ? 1 : 0;
            const down = s.pen === 'down' ? 1 : 0;
            const end = s.pen === 'end' ? 1 : 0;
            return [s.dx, s.dy, down, up, end];
        });
        return callCallback(this.generateInternal(options, strokes), callback);
    }

  /**
   * reset the model to its original state
   */
  reset() {
        this.penState = this.model.zeroInput();
        if (this.rnnState) {
            this.rnnState = this.model.zeroState();
        }
    }
}

/**
 * @param {string} model - Required. The name of the sketch model to be loaded.
 *    The names can be found in the models.js file.
 *    Can also provide a path to a model file in '.gen.json' format.
 * @param {function} [callback] - Optional. A callback function that is called once the model has loaded.
 *    If no callback is provided, it will return a promise that will be resolved once the model has loaded.
 * @param {boolean} [large] // TODO: accept options object - this is a breaking change so I need to check that it's ok
 * @return {SketchRNN}
 */
const sketchRNN = (model, callback, large = true) => new SketchRNN(model, {large}, callback);

export default sketchRNN;
