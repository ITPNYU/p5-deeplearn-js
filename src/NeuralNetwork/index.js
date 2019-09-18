// Copyright (c) 2019 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/*
Generic NeuralNetwork class
*/

import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import callCallback from '../utils/callcallback';
import {
  saveBlob
} from '../utils/io';


const DEFAULTS = {
  task: 'regression',
  // activation: 'sigmoid',
  activationHidden: 'sigmoid',
  activationOutput: 'sigmoid',
  debug: true,
  learningRate: 0.25,
  inputUnits: 2,
  outputUnits: 1,
  noVal: null,
  hiddenUnits: 1,
  modelMetrics: ['accuracy'],
  modelLoss: 'meanSquaredError',
  modelOptimizer: null,
  batchSize: 64,
  epochs: 32,
  inputKeys: [],
  outputKeys: []
}

class NeuralNetwork {
  /**
   * Create a Neural Network.
   * @param {object} options - An object with options.
   */
  constructor(options) {
    // TODO: create the model based on many more options and defaults

    this.config = {
      task: options.task || DEFAULTS.task,
      debug: options.debug || DEFAULTS.debug,
      // activation: options.activation || DEFAULTS.activation,
      activationHidden: options.activationHidden || DEFAULTS.activationHidden,
      activationOutput: options.activationOutput || DEFAULTS.activationOutput,
      inputUnits: options.inputs || DEFAULTS.inputUnits,
      outputUnits: options.outputs || DEFAULTS.outputUnits,
      noVal: options.noVal || options.outputs,
      hiddenUnits: options.hiddenUnits || DEFAULTS.hiddenUnits,
      learningRate: options.outputs || DEFAULTS.learningRate,
      modelMetrics: options.modelMetrics || DEFAULTS.modelMetrics,
      modelLoss: options.modelLoss || DEFAULTS.modelLoss,
      modelOptimizer: options.modelOptimizer || DEFAULTS.modelOptimizer,
      batchSize: options.batchSize || DEFAULTS.batchSize,
      epochs: options.epochs || DEFAULTS.epochs,
    }

    this.model = this.createModel();
    this.data = {
      xs: [],
      ys: []
    }

  }

  /**
   * createModel()
   * Depending on the task -- classification or regression -- returns a simple model architecture
   */
  createModel() {

    switch (this.config.task) {
      case 'regression':
        this.config.modelOptimizer = tf.train.sgd(this.config.learningRate);
        return this.createModelInternal();
      case 'classification':

        // Change the default activations for classifications
        this.config.hiddenUnits = 16;
        this.config.activationHidden = 'relu' // 'relu',
        this.config.activationOutput = 'softmax' // 'relu',
        this.config.modelLoss = 'categoricalCrossentropy'
        this.config.modelOptimizer = tf.train.adam();

        return this.createModelInternal();
      default:
        console.log('no model exists for this type of task yet!');
        return tf.sequential();
    }
  }

  /**
   * createModelInternal()
   * Creates a sequential model with 1 hidden layer, and 1 output layer
   */
  createModelInternal() {
    const model = tf.sequential();

    const hidden = tf.layers.dense({
      units: this.config.hiddenUnits,
      inputShape: [this.config.inputUnits],
      activation: this.config.activationHidden,
    });

    // TODO: figure out if we want to add in the ability to add more layers?

    const output = tf.layers.dense({
      units: this.config.outputUnits,
      activation: this.config.activationOutput,
    });

    model.add(hidden);
    model.add(output);

    model.compile({
      optimizer: this.config.modelOptimizer,
      loss: this.config.modelLoss,
      metrics: this.config.modelMetrics,
    });

    if (this.config.debug) {
      tfvis.show.modelSummary({
        name: 'Model Summary'
      }, model);
    }

    return model;
  }


  /**
   * Loads in CSV data by URL
   * @param {*} options or DATAURL
   * @param {*} callback
   */
  loadData(optionsOrDataUrl, callback) {
    let options;
    if (typeof optionsOrDataUrl === 'string') {
      options = {
        dataUrl: optionsOrDataUrl
      }
    } else {
      options = optionsOrDataUrl;
    }
    return callCallback(this.loadDataInternal(options), callback);
  }

  // TODO: need to add loading in for JSON data
  /**
   * Loads in a CSV file
   * @param {*} options
   */
  async loadDataInternal(options) {

    this.config.inputKeys = options.inputKeys || [...new Array(this.inputUnits).fill(null).map((v, idx) => idx)];
    this.config.outputKeys = options.outputKeys || [...new Array(this.outputUnits / 2).fill(null).map((v, idx) => idx)];

    const outputLabel = this.config.outputKeys[0];
    const inputLabels = this.config.inputKeys;

    let data = tf.data.csv(options.dataUrl, {
      columnConfigs: {
        [outputLabel]: {
          isLabel: true
        }
      }
    });

    data = await data.toArray();

    if (this.config.debug) {
      const values = inputLabels.map(label => {
        return data.map(item => {
          return {
            x: item.xs[label],
            y: item.ys[outputLabel]
          }
        })
      })

      tfvis.render.scatterplot({
        name: 'debug mode'
      }, {
        values
      }, {
        xLabel: 'X',
        yLabel: 'Y',
        height: 300
      });
    }

    return data;
  }

  /* eslint class-methods-use-this: ["error", { "exceptMethods": ["shuffle"] }] */
  shuffle(data){
      tf.util.shuffle(data);
  }


  normalize(data) {
    return tf.tidy(() => {
      const outputLabel = this.config.outputKeys[0];
      const inputLabels = this.config.inputKeys;

      // TODO: need to test this for regression data.

      // Step 2. Convert data to Tensor
      // const inputs = data.map(d => inputLabels.map(header => d.xs[header]));
      const inputs = inputLabels.map(header => data.map(d => d.xs[header]))
      const targets = data.map(d => d.ys[outputLabel]);

      const inputTensor = tf.tensor(inputs);

      let outputTensor;
      if (this.config.task === 'classification') {
        outputTensor = tf.oneHot(tf.tensor1d(targets, 'int32'), this.config.noVal);
      } else {
        outputTensor = tf.tensor(targets);
      }


      // // Step 3. Normalize the data to the range 0 - 1 using min-max scaling
      const inputMax = inputTensor.max();
      const inputMin = inputTensor.min();
      const targetMax = outputTensor.max();
      const targetMin = outputTensor.min();

      const normalizedInputs = inputTensor.sub(targetMin).div(targetMax.sub(inputMin)).flatten().reshape([data.length, this.config.inputUnits]);

      // console.log()
      const normalizedOutputs = outputTensor.sub(targetMin).div(targetMax.sub(targetMin));

      inputTensor.max(1).print();

      return {
        inputs: normalizedInputs, // normalizedInputs,
        targets: normalizedOutputs,
        // Return the min/max bounds so we can use them later.
        inputMax,
        inputMin,
        targetMax,
        targetMin,
      }
    });
  }

  addData(xs, ys) {
    this.training.xs.push(xs);
    this.training.ys.push(ys);
  }

  train(inputs, labels, callback) {
    return callCallback(this.trainInternal(inputs, labels), callback);
  }

  async trainInternal(inputs, labels) {
    const { batchSize, epochs } = this.config;
    let xs;
    let ys;

    // check if the inputs are tensors, if not, convert!
    if (!(inputs instanceof tf.Tensor)) {
      xs = tf.tensor(inputs)
      ys = tf.tensor(labels)
    } else {
      xs = inputs;
      ys = labels;
    }

    await this.model.fit(xs, ys, {
      shuffle: true,
      batchSize,
      epochs,
      validationSplit: 0.1,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          console.log(`Epoch: ${epoch} - accuracy: ${logs.loss.toFixed(3)}`);
        },
        onTrainEnd: () => {
          console.log(`training complete!`);
        }
      },
    });
    xs.dispose();
    ys.dispose();
  }

  

  predict(input, callback) {
    return callCallback(this.predictInternal(input), callback);
  }


  async predictInternal(sample) {
    const xs = tf.tensor(sample, [1, sample.length]);
    const ys = this.model.predict(xs);

    const results = {
      output: await ys.data(),
      tensor: ys
    }
    xs.dispose();

    return results;
  }

  async save(callback, name) {
    this.model.save(tf.io.withSaveHandler(async (data) => {
      let modelName = 'model';
      if (name) modelName = name;

      this.weightsManifest = {
        modelTopology: data.modelTopology,
        weightsManifest: [{
          paths: [`./${modelName}.weights.bin`],
          weights: data.weightSpecs,
        }]
      };
      await saveBlob(data.weightData, `${modelName}.weights.bin`, 'application/octet-stream');
      await saveBlob(JSON.stringify(this.weightsManifest), `${modelName}.json`, 'text/plain');
      if (callback) {
        callback();
      }
    }));
  }

  async load(filesOrPath = null, callback) {
    if (typeof filesOrPath !== 'string') {
      let model = null;
      let weights = null;
      Array.from(filesOrPath).forEach((file) => {
        if (file.name.includes('.json')) {
          model = file;
          const fr = new FileReader();
          fr.readAsText(file);
        } else if (file.name.includes('.bin')) {
          weights = file;
        }
      });
      this.model = await tf.loadLayersModel(tf.io.browserFiles([model, weights]));
    } else {
      fetch(filesOrPath)
        .then(r => r.json());
      this.model = await tf.loadLayersModel(filesOrPath);
    }
    if (callback) {
      callback();
    }
    return this.model;
  }
}

const neuralNetwork = (inputsOrOptions, outputs) => {
  let options;
  if (inputsOrOptions instanceof Object) {
    options = inputsOrOptions;
  } else {
    options = {
      input: inputsOrOptions,
      outputs,
    };
  }

  const instance = new NeuralNetwork(options);
  return instance;
};

export default neuralNetwork;