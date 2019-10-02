// Copyright (c) 2019 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/*
Generic NeuralNetwork class
*/

import * as tf from '@tensorflow/tfjs';
import callCallback from '../utils/callcallback';
// import {
//   saveBlob
// } from '../utils/io';
// import { input } from '@tensorflow/tfjs';
import DEFAULTS from './NeuralNetworkDefaults';
import NeuralNetworkData from './NeuralNetworkData';
import NeuralNetworkVis from './NeuralNetworkVis';

class NeuralNetwork {
  /**
   * Create a Neural Network.
   * @param {object} options - An object with options.
   */
  constructor(options, callback) {
    // model config
    this.config = {
      // debugging
      debug: options.debug || DEFAULTS.debug,
      // architecture
      architecture: {
        task: options.task || DEFAULTS.task,
        // array of layers, the last is always the output layer
        layers: [],
        // array of activations corresponding to the layer number
        activations: [],
        // hiddenUnits
        hiddenUnits: options.hiddenUnits || DEFAULTS.hiddenUnits,
        // the units of the layers will come from the config.dataOptions
      },
      training: {
        // defined either on instantiation or in .train(options)
        batchSize: options.batchSize || DEFAULTS.batchSize,
        epochs: options.epochs || DEFAULTS.epochs,
        // will depend on the config.architecture.task
        learningRate: options.learningRate || DEFAULTS.learningRate,
        modelMetrics: options.modelMetrics || DEFAULTS.modelMetrics,
        modelLoss: options.modelLoss || DEFAULTS.modelLoss,
        modelOptimizer: options.modelOptimizer || DEFAULTS.modelOptimizer,
      },
      // data 
      dataOptions: {
        dataUrl: options.dataUrl || null,
        inputs: options.inputs || DEFAULTS.inputs,
        outputs: options.outputs || DEFAULTS.outputs,
        // TODO: adding option for normalization
        normalizationOptions: options.normalizationOptions || null
      },

    }


    // TODO: maybe we create a set of configs for 
    // regression vs. classification
    // set the default activations:
    if (this.config.architecture.task === 'regression') {
      // current defaults are for regression
      const activationHidden = options.activationHidden || DEFAULTS.activationHidden;
      const activationOutput = options.activationOutput || DEFAULTS.activationOutput;

      this.config.training.modelOptimizer = options.modelOptimizer || tf.train.adam(this.config.training.learningRate);
      this.config.architecture.activations = [activationHidden, activationOutput];
    } else if (this.config.architecture.task === 'classification') {
      // set classification specs different from regression in DEFAULTS
      const activationHidden = options.activationHidden || DEFAULTS.activationHidden;
      const activationOutput = options.activationOutput || 'softmax';

      this.config.architecture.activations = [activationHidden, activationOutput];
      this.config.training.modelLoss = options.modelLoss || 'categoricalCrossentropy';
      this.config.training.modelOptimizer = options.modelOptimizer || tf.train.sgd(this.config.training.learningRate);
    } else {
      console.log(`task not defined. please set task: classification OR regression`);
    }

    // vis class
    this.vis = new NeuralNetworkVis();
    // data class
    this.data = new NeuralNetworkData(this.config);
    // check if the model is ready
    this.ready = false;
    // the model
    this.model = null;

    // initialize
    this.init(callback);

    // console.log(typeof callback, typeof tf, typeof callCallback)

  }

  /**
   * ----------------------------------------
   * --- model creation / initialization ----
   * ---------------------------------------- 
   */

  /**
   * Initialize the model creation
   */
  init(callback) {
    // Create the model based on data or the inputs/outputs
    if (this.config.dataOptions.dataUrl !== null) {
      this.ready = this.createModelFromData(callback);
    } else {

      this.data.meta.inputUnits = this.config.dataOptions.inputs;
      this.data.meta.outputUnits = this.config.dataOptions.outputs;

      // convert the input number to an array of keys e.g. [label1, label2, label3]
      this.data.config.dataOptions.inputs = this.data.createNamedIO(this.data.config.dataOptions.inputs, 'input');
      this.data.config.dataOptions.outputs = this.data.createNamedIO(this.data.config.dataOptions.outputs, 'output');

      this.model = this.createModel();
      this.ready = true;
    }
  }

  /**
   * create Model
   */
  createModel() {

    switch (this.config.architecture.task) {
      case 'regression':
        // if the layers are not defined default to a 
        // neuralnet with 2 layers
        this.defineModelLayers();
        return this.createModelInternal();
      case 'classification':
        // if the layers are not defined default to a 
        // neuralnet with 2 layers
        this.defineModelLayers();
        return this.createModelInternal();
      default:
        console.log('no model exists for this type of task yet!');
        return tf.sequential();
    }
  }

  /**
   * Define the model layers
   */
  defineModelLayers() {
    if (!this.config.architecture.layers.length > 0) {
      this.config.architecture.layers = [];

      const {
        activations,
        hiddenUnits
      } = this.config.architecture;

      const hidden = tf.layers.dense({
        units: hiddenUnits,
        inputShape: [this.data.meta.inputUnits],
        activation: activations[0],
      });

      const output = tf.layers.dense({
        units: this.data.meta.outputUnits,
        activation: activations[1],
      });

      this.config.architecture.layers = [hidden, output];

    }
  }


  createModelInternal() {
    const model = tf.sequential();

    // add the layers to the model as defined in config.architecture.layers
    this.config.architecture.layers.forEach(layer => {
      model.add(layer);
    });

    // compile the model
    const {
      modelOptimizer,
      modelLoss,
      modelMetrics
    } = this.config.training;

    model.compile({
      optimizer: modelOptimizer,
      loss: modelLoss,
      metrics: modelMetrics,
    });

    return model;
  }

  /**
   * create model from data
   * @param {*} callback 
   */
  createModelFromData(callback) {
    return callCallback(this.createModelFromDataInternal(), callback)
  }

  /**
   * Creates model architecture from the loaded data
   */
  async createModelFromDataInternal() {
    // load the data
    await this.data.loadData();
    // check the input columns for data type to
    // calculate the total number of inputs
    // and outputs
    this.data.getIOUnits();
    // create the model
    this.model = this.createModel();
  }

  /**
   * ----------------------------------------
   * ----- adding data / training -----------
   * ---------------------------------------- 
   */
  /**
   * Adds an endpoint to call data.addData()
   * @param {*} xs 
   * @param {*} ys 
   */
  addData(xs, ys) {
    this.data.addData(xs, ys);
  }

  /**
   * normalize the data.raw
   */
  normalize() {
    this.data.normalize();
  }

  /**
   * User-facing neural network training
   * @param {*} optionsOrCallback
   * @param {*} callback
   */
  train(optionsOrCallback, optionsOrWhileTraining, callback) {
    let options;
    let whileTrainingCb;
    let finishedTrainingCb;
    if (typeof optionsOrCallback === 'object' &&
      typeof optionsOrWhileTraining === 'function' &&
      typeof callback === 'function'
    ) {
      options = optionsOrCallback;
      whileTrainingCb = optionsOrWhileTraining;
      finishedTrainingCb = callback;
    } else if (typeof optionsOrCallback === 'object' &&
      typeof optionsOrWhileTraining === 'function') {
      options = optionsOrCallback;
      whileTrainingCb = null;
      finishedTrainingCb = optionsOrWhileTraining;
    } else if (typeof optionsOrCallback === 'function' &&
      typeof optionsOrWhileTraining === 'function'
    ) {
      options = {};
      whileTrainingCb = optionsOrCallback;
      finishedTrainingCb = optionsOrWhileTraining;
    } else {
      options = {};
      whileTrainingCb = null;
      finishedTrainingCb = optionsOrCallback;
    }

    return callCallback(this.trainInternal(options, whileTrainingCb), finishedTrainingCb);
  }

  /**
   * Train the neural network
   * @param {*} options
   */
  async trainInternal(options, whileTrainingCallback) {
    const batchSize = options.batchSize || this.config.batchSize;
    const epochs = options.epochs || this.config.epochs;

    const whileTraining = (typeof whileTrainingCallback === 'function') ?
      whileTrainingCallback : (epoch, logs) => console.log(`Epoch: ${epoch} - accuracy: ${logs.loss.toFixed(3)}`);

    let xs;
    let ys;

    const {
      inputs,
      outputs
    } = this.data.data.tensor;

    // check if the inputs are tensors, if not, convert!
    if (!(inputs instanceof tf.Tensor)) {
      xs = tf.tensor(inputs)
      ys = tf.tensor(outputs)
    } else {
      xs = inputs;
      ys = outputs;
    }

    let modelFitCallbacks;
    if (this.config.debug) {
      modelFitCallbacks = [
        this.vis.trainingVis(),
        {
          onEpochEnd: whileTraining
        }
      ]
    } else {
      modelFitCallbacks = [{
        onEpochEnd: whileTraining
      }]
    }

    await this.model.fit(xs, ys, {
      shuffle: true,
      batchSize,
      epochs,
      validationSplit: 0.1,
      callbacks: modelFitCallbacks
    });
    xs.dispose();
    ys.dispose();
  }


  /**
   * ----------------------------------------
   * ----- prediction / classification-------
   * ---------------------------------------- 
   */
   /**
   * Classify()
   * Runs the classification if the neural network is doing a
   * classification task
   * @param {*} input
   * @param {*} callback
   */
  classify(input, callback) {
    return callCallback(this.predictInternal(input), callback);
  }

  /**
   * Userfacing prediction function
   * @param {*} input
   * @param {*} callback
   */
  predict(input, callback) {
    return callCallback(this.predictInternal(input), callback);
  }

  /**
   * Make a prediction based on the given input
   * @param {*} sample
   */
  async predictInternal(sample){
    // 1. Handle the input sample
    // either an array of values in order of the inputs
    // OR an JSON object of key/values
    let inputData = [];
    if (sample instanceof Array) {
      inputData = sample;
    } else if (sample instanceof Object) {
      // TODO: make sure that the input order is preserved!
      const headers = this.data.config.dataOptions.inputs;
      inputData = headers.map(prop => {
        return sample[prop]
      });
    }

    // 2. onehot encode the sample if necessary
    let encodedInput = [];
    
    Object.entries(this.data.meta.inputs).forEach( (arr) => {
      const prop = arr[0];
      const {dtype} = arr[1];

      // to ensure that we get the value in the right order
      const valIndex = this.data.config.dataOptions.inputs.indexOf(prop);
      const val = inputData[valIndex];

      if(dtype === 'number'){
        const {inputMin, inputMax} = this.data.data;
        const normVal = (val - inputMin[valIndex]) / (inputMax[valIndex] - inputMin[valIndex]);
        encodedInput.push(normVal);
      } else if (dtype === 'string'){
        const {legend} = arr[1];
        const onehotVal = legend[val]
        encodedInput = [...encodedInput, ...onehotVal]
      }

    })

    const xs = tf.tensor(encodedInput, [1, this.data.meta.inputUnits]);
    const ys = this.model.predict(xs);
    
    const results = {
      output: null,
      tensor: null,
    };

    if(this.config.architecture.task === 'classification'){
      const predictions = await ys.data();
      // TODO: Check to see if this fails with numeric values
      // since no legend exists
      const outputData = Object.entries(this.data.meta.outputs).map( (arr) => {
        const {legend} = arr[1];
        // TODO: the order of the legend items matters
        // Likey this means instead of `.push()`,
        // we should do .unshift()
        // alternatively we can use 'reverse()' here.
        return Object.entries(legend).map( (legendArr, idx) => {
          const prop = legendArr[0];
          return {
            label: prop,
            confidence: predictions[idx]
          }
        }).sort((a, b) => b.confidence - a.confidence);
      })[0];

      // console.log(predictions);
      results.output =  outputData;
      results.tensor = ys;

    } else if (this.config.architecture.task === 'regression') {
      const predictions = await ys.data();
      
      const outputData = Object.entries(this.data.meta.outputs).map((item, idx) => {
        const {outputMin, outputMax} = this.data.data;
        const val = (predictions[idx] * (outputMax[idx] - outputMin[idx])) + outputMin[idx];
        return {
          value: val
        }
      })[0];

      results.output =  outputData;
      results.tensor = ys;
    }

    xs.dispose();    
    return results;
    

  }

}




/**
 * Create an instance of the NeuralNetwork
 * @param {*} inputsOrOptions
 * @param {*} outputsOrCallback
 * @param {*} callback
 */
const neuralNetwork = (inputsOrOptions, outputsOrCallback, callback) => {

  let options;
  let cb;

  if (inputsOrOptions instanceof Object) {
    options = inputsOrOptions;
    cb = outputsOrCallback;
  } else {
    options = {
      inputs: inputsOrOptions,
      outputs: outputsOrCallback,
    };
    cb = callback;
  }

  const instance = new NeuralNetwork(options, cb);
  return instance;
};

export default neuralNetwork;