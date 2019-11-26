import * as tf from '@tensorflow/tfjs';
import NeuralNetwork from './NeuralNetwork';
import NeuralNetworkData from './NeuralNetworkData';
import callCallback from '../utils/callcallback';

const DEFAULTS = {
  inputs: [],
  outputs: [],
  dataUrl: null,
  task: null
}
class DiyNeuralNetwork {

  constructor(options, cb) {
    this.callback = cb;
    this.options = {
      ...DEFAULTS,
      ...options
    } || DEFAULTS;

    this.neuralNetwork = new NeuralNetwork();
    this.neuralNetworkData = new NeuralNetworkData();

    this.data = {
      training: []
    }


    this.ready = false;
    this.init(this.callback);

  }

  /**
   * init
   * @param {*} callback 
   */
  init(callback) {
    if (this.options.dataUrl !== null) {
      this.ready = this.loadData(this.options, callback);
    } else {
      this.ready = true;
    }
  }


  /**
   * loadData
   * @param {*} options 
   * @param {*} callback 
   */
  loadData(options, callback) {
    return callCallback(this.loadDataInternal(options), callback)
  }

  /**
   * loadDataInternal
   * @param {*} options 
   */
  async loadDataInternal(options) {
    const {
      dataUrl,
      inputs,
      outputs
    } = options;

    if (dataUrl.endsWith('.csv')) {
      await this.neuralNetworkData.loadCSV(dataUrl, inputs, outputs);
    } else if (dataUrl.endsWith('.json')) {
      await this.neuralNetworkData.loadJSON(dataUrl, inputs, outputs);
    } else if (dataUrl.includes('blob')) {
      await this.neuralNetworkData.loadBlob(dataUrl, inputs, outputs);
    } else {
      console.log('Not a valid data format. Must be csv or json')
    }

    // once the data are loaded, create the metadata 
    // and prep the data for training
    this.createMetaDataFromData();
    this.warmUp();
  }

  /**
   * createMetaDataFromData
   * create your meta data about your data
   * @param {*} _dataRaw 
   */
  createMetaDataFromData(_dataRaw = null) {
    const dataRaw = _dataRaw === null ? this.neuralNetworkData.data.raw : _dataRaw;

    const meta = this.neuralNetworkData.createMetaDataFromData(dataRaw)
    this.neuralNetworkData.meta = meta;
    return meta;
  }



  /**
   * summarizeData
   * adds min and max to the meta of each input and output property
   */
  summarizeData(_dataRaw = null, _meta = null) {
    const dataRaw = _dataRaw === null ? this.neuralNetworkData.data.raw : _dataRaw;
    const meta = _meta === null ? this.neuralNetworkData.meta : _meta;

    const inputMeta = this.neuralNetworkData.getRawStats(dataRaw, meta.inputs, 'xs');
    const outputMeta = this.neuralNetworkData.getRawStats(dataRaw, meta.outputs, 'ys');

    this.neuralNetworkData.meta.inputs = inputMeta;
    this.neuralNetworkData.meta.outputs = outputMeta;

    return this.neuralNetworkData.meta;
  }

  /**
   * warmUp
   * @param {*} _dataRaw 
   * @param {*} _meta 
   */
  warmUp(_dataRaw = null, _meta = null) {
    const dataRaw = _dataRaw === null ? this.neuralNetworkData.data.raw : _dataRaw;
    const meta = _meta === null ? this.neuralNetworkData.meta : _meta;

    // summarize data
    const updatedMeta = this.summarizeData(dataRaw, meta);
    // apply one hot encodings
    const encodedData = this.neuralNetworkData.applyOneHotEncodingsToDataRaw(dataRaw, meta);


    // set this equal to the training data
    this.data.training = encodedData;

    return {
      meta: updatedMeta,
      data: {
        raw: encodedData
      }
    }
  }


  /**
   * convertTrainingDataToTensors
   * @param {*} _trainingData 
   * @param {*} _meta 
   */
  convertTrainingDataToTensors(_trainingData = null, _meta = null) {
    const trainingData = _trainingData === null ? this.data.training : _trainingData;
    const meta = _meta === null ? this.neuralNetworkData.meta : _meta;

    return this.neuralNetworkData.convertRawToTensors(trainingData, meta);
  }

  /**
   * normalizeData
   * @param {*} _dataRaw 
   * @param {*} _meta 
   */
  normalizeData(_dataRaw = null, _meta = null) {
    const dataRaw = _dataRaw === null ? this.neuralNetworkData.data.raw : _dataRaw;
    const meta = _meta === null ? this.neuralNetworkData.meta : _meta;

    const normalizedInputs = this.neuralNetworkData.normalizeRaws(dataRaw, meta.inputs, 'xs');
    const normalizedOutputs = this.neuralNetworkData.normalizeRaws(dataRaw, meta.outputs, 'ys');
    const trainingData = this.neuralNetworkData.zipArrays(normalizedInputs, normalizedOutputs)

    // set this equal to the training data
    this.data.training = trainingData;

    return trainingData;
  }

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

    this.trainInternal(options, whileTrainingCb, finishedTrainingCb);
  }

  /**
   * train
   * @param {*} _options 
   * @param {*} _cb 
   */
  trainInternal(_options, whileTrainingCb, finishedTrainingCb) {

    const options = {
      epochs: 10,
      batchSize: 32,
      validationSplit: 0.1,
      whileTraining: null,
      compile: true,
      ..._options
    };

    options.whileTraining = whileTrainingCb === null ?
      (epoch, loss) => {
        console.log(epoch, loss.loss)
      } : whileTrainingCb;

    // if inputs and outputs are not specified
    // in the options, then create the tensors
    // from the this.neuralNetworkData.data.raws
    if (!options.inputs && !options.outputs) {
      const {
        inputs,
        outputs
      } = this.convertTrainingDataToTensors();
      options.inputs = inputs;
      options.outputs = outputs;
    }

    // if the model does not have any layers defined yet
    // then use the default structure
    if (!this.neuralNetwork.isLayered) {
      this.addDefaultLayers(this.options.task);
    }

    if (options.compile) {
      // compile the model with defaults
      this.compile()
    }

    // train once the model is compiled
    this.neuralNetwork.train(options, finishedTrainingCb);
  }


  addDefaultLayers(_task) {
    const {
      inputUnits,
      outputUnits
    } = this.neuralNetworkData.meta
    switch (_task.toLowerCase()) {
      case 'classification':

        this.addLayer(this.createDenseLayer({
          inputShape: [inputUnits]
        }));

        this.addLayer(this.createDenseLayer({
          units: outputUnits,
          activation: 'softmax'
        }))

        break;
      case 'regression':
        this.addLayer(
          this.createDenseLayer({
            inputShape: [inputUnits]
          }))
        this.addLayer(this.createDenseLayer({
          units: outputUnits,
          activation: 'sigmoid'
        }))
        break;
      default:
        console.log('no imputUnits or outputUnits defined')
        break;
    }
  }


  /**
   * 
   * @param {*} _options 
   */
  compile(_modelOptions = null, _learningRate = null) {
    const LEARNING_RATE = _learningRate === null ? 0.25 : _learningRate;

    let options = {};

    if (_modelOptions !== null) {
      options = {
        ..._modelOptions
      }
    } else if (this.options.task === 'classification') {
      options = {
        loss: 'categoricalCrossentropy',
        optimizer: tf.train.sgd,
        metrics: ['accuracy'],
      }
    } else if (this.options.task === 'regression') {
      options = {
        loss: 'meanSquaredError',
        optimizer: tf.train.adam,
        metrics: ['accuracy'],
      }
    }

    options.optimizer = options.optimizer ?
      this.neuralNetwork.setOptimizerFunction(LEARNING_RATE, options.optimizer) :
      this.neuralNetwork.setOptimizerFunction(LEARNING_RATE, tf.train.sgd)

    this.neuralNetwork.compile(options);
  }


  predict(_input, _cb) {

    let inputData = [];
    if (_input instanceof Array) {
      inputData = _input;
    } else if (_input instanceof Object) {
      // TODO: make sure that the input order is preserved!
      const headers = Object.keys(this.neuralNetworkData.meta.inputs);
      inputData = headers.map(prop => {
        return _input[prop]
      });
    }

    inputData = tf.tensor([inputData])
    this.neuralNetwork.predict(inputData, this.neuralNetwork.meta, _cb)
  }

  classify(_input, _cb) {
    let inputData = [];
    if (_input instanceof Array) {
      inputData = _input;
    } else if (_input instanceof Object) {
      // TODO: make sure that the input order is preserved!
      const headers = Object.keys(this.neuralNetworkData.meta.inputs);
      inputData = headers.map(prop => {
        return _input[prop]
      });
    }

    inputData = tf.tensor([inputData])
    this.neuralNetwork.classify(inputData, this.neuralNetworkData.meta, _cb);
  }


  /**
   * addLayer
   * @param {*} _options 
   */
  addLayer(_options) {
    this.neuralNetwork.addLayer(_options);
  }


  /**
   * createDenseLayer
   * @param {*} _options 
   */
  // eslint-disable-next-line class-methods-use-this
  createDenseLayer(_options) {

    const options = Object.assign({}, {
      units: 16,
      activation: 'relu',
      ..._options
    });

    return tf.layers.dense(options);
  }

  /**
   * createConv2dLayer
   * @param {*} _options 
   */
  // eslint-disable-next-line class-methods-use-this
  createConv2dLayer(_options) {
    const options = Object.assign({}, {
      kernelSize: 5,
      filters: 8,
      strides: 1,
      activation: 'relu',
      kernelInitializer: 'varianceScaling',
      ..._options
    })

    return tf.layers.conv2d(options);
  }



}


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

  const instance = new DiyNeuralNetwork(options, cb);
  return instance;
}

export default neuralNetwork