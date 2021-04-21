import * as tf from '@tensorflow/tfjs';
import axios from 'axios';
import { saveBlob } from '../utils/io';
import nnUtils from './NeuralNetworkUtils';

class NeuralNetworkData {
  constructor() {
    this.meta = {
      inputUnits: null, // Number
      outputUnits: null, // Number
      // objects describing input/output data by property name
      inputs: {}, // { name1: {dtype}, name2: {dtype}  }
      outputs: {}, // { name1: {dtype} }
      isNormalized: false, // Boolean - keep this in meta for model saving/loading
    };

    this.isMetadataReady = false;
    this.isWarmedUp = false;

    this.data = {
      raw: [], // array of {xs:{}, ys:{}}
    };

    // methods
    // summarize data
    this.createMetadata = this.createMetadata.bind(this);
    this.getDataStats = this.getDataStats.bind(this);
    this.getInputMetaStats = this.getInputMetaStats.bind(this);
    this.getDataUnits = this.getDataUnits.bind(this);
    this.getInputMetaUnits = this.getInputMetaUnits.bind(this);
    this.getDTypesFromData = this.getDTypesFromData.bind(this);
    // add data
    this.addData = this.addData.bind(this);
    // data conversion
    this.convertRawToTensors = this.convertRawToTensors.bind(this);
    // data normalization / unnormalization
    this.normalizeDataRaw = this.normalizeDataRaw.bind(this);
    this.normalizeInputData = this.normalizeInputData.bind(this);
    this.normalizeArray = this.normalizeArray.bind(this);
    this.unnormalizeArray = this.unnormalizeArray.bind(this);
    // one hot
    this.applyOneHotEncodingsToDataRaw = this.applyOneHotEncodingsToDataRaw.bind(this);
    this.getDataOneHot = this.getDataOneHot.bind(this);
    this.getInputMetaOneHot = this.getInputMetaOneHot.bind(this);
    this.createOneHotEncodings = this.createOneHotEncodings.bind(this);
    // Saving / loading data
    this.loadDataFromUrl = this.loadDataFromUrl.bind(this);
    this.loadJSON = this.loadJSON.bind(this);
    this.loadCSV = this.loadCSV.bind(this);
    this.loadBlob = this.loadBlob.bind(this);
    this.loadData = this.loadData.bind(this);
    this.saveData = this.saveData.bind(this);
    this.saveMeta = this.saveMeta.bind(this);
    this.loadMeta = this.loadMeta.bind(this);
    // data loading helpers
    this.findEntries = this.findEntries.bind(this);
    this.formatRawData = this.formatRawData.bind(this);
    this.csvToJSON = this.csvToJSON.bind(this);
  }

  /**
   * ////////////////////////////////////////////////////////
   * Summarize Data
   * ////////////////////////////////////////////////////////
   */

  /**
   * create the metadata from the data
   * this covers:
   *  1. getting the datatype from the data
   *  2. getting the min and max from the data
   *  3. getting the oneHot encoded values
   *  4. getting the inputShape and outputUnits from the data
   * @param {*} dataRaw
   * @param {*} inputShape
   */
  createMetadata(dataRaw, inputShape = null) {
    // get the data type for each property
    this.getDTypesFromData(dataRaw);
    // get the stats - min, max
    this.getDataStats(dataRaw);
    // onehot encode
    this.getDataOneHot(dataRaw);
    // calculate the input units from the data
    this.getDataUnits(dataRaw, inputShape);

    this.isMetadataReady = true;
    return { ...this.meta };
  }

  /*
   * ////////////////////////////////////////////////
   * data Summary
   * ////////////////////////////////////////////////
   */

  /**
   * get stats about the data
   * @param {*} dataRaw
   */
  getDataStats(dataRaw) {
    const meta = Object.assign({}, this.meta);

    const inputMeta = this.getInputMetaStats(dataRaw, meta.inputs, 'xs');
    const outputMeta = this.getInputMetaStats(dataRaw, meta.outputs, 'ys');

    meta.inputs = inputMeta;
    meta.outputs = outputMeta;

    this.meta = {
      ...this.meta,
      ...meta,
    };

    return meta;
  }

  /**
   * getRawStats
   * get back the min and max of each label
   * @param {*} dataRaw
   * @param {*} inputOrOutputMeta
   * @param {*} xsOrYs
   */
  // eslint-disable-next-line no-unused-vars, class-methods-use-this
  getInputMetaStats(dataRaw, inputOrOutputMeta, xsOrYs) {
    const inputMeta = Object.assign({}, inputOrOutputMeta);

    Object.keys(inputMeta).forEach(k => {
      if (inputMeta[k].dtype === 'string') {
        inputMeta[k].min = 0;
        inputMeta[k].max = 1;
      } else if (inputMeta[k].dtype === 'number') {
        const dataAsArray = dataRaw.map(item => item[xsOrYs][k]);
        inputMeta[k].min = nnUtils.getMin(dataAsArray);
        inputMeta[k].max = nnUtils.getMax(dataAsArray);
      } else if (inputMeta[k].dtype === 'array') {
        const dataAsArray = dataRaw.map(item => item[xsOrYs][k]).flat();
        inputMeta[k].min = nnUtils.getMin(dataAsArray);
        inputMeta[k].max = nnUtils.getMax(dataAsArray);
      }
    });

    return inputMeta;
  }

  /**
   * get the data units, inputshape and output units
   * @param {*} dataRaw
   */
  getDataUnits(dataRaw, _arrayShape = null) {
    const arrayShape = _arrayShape !== null ? _arrayShape : undefined;
    const meta = Object.assign({}, this.meta);

    // if the data has a shape pass it in
    let inputShape;
    if (arrayShape) {
      inputShape = arrayShape;
    } else {
      inputShape = [this.getInputMetaUnits(dataRaw, meta.inputs)].flat();
    }

    const outputShape = this.getInputMetaUnits(dataRaw, meta.outputs);

    meta.inputUnits = inputShape;
    meta.outputUnits = outputShape;

    this.meta = {
      ...this.meta,
      ...meta,
    };

    return meta;
  }

  /**
   * get input
   * @param {*} _inputsMeta
   * @param {*} _dataRaw
   */
  // eslint-disable-next-line class-methods-use-this, no-unused-vars
  getInputMetaUnits(_dataRaw, _inputsMeta) {
    let units = 0;
    const inputsMeta = Object.assign({}, _inputsMeta);

    Object.entries(inputsMeta).forEach(arr => {
      const { dtype } = arr[1];
      if (dtype === 'number') {
        units += 1;
      } else if (dtype === 'string') {
        const { uniqueValues } = arr[1];

        const uniqueCount = uniqueValues.length;
        units += uniqueCount;
      } else if (dtype === 'array') {
        // TODO: User must input the shape of the
        // image size correctly.
        units = [];
      }
    });

    return units;
  }

  /**
   * getDTypesFromData
   * gets the data types of the data we're using
   * important for handling oneHot
   */
  getDTypesFromData(_dataRaw) {
    const meta = {
      ...this.meta,
      inputs: {},
      outputs: {},
    };

    const sample = _dataRaw[0];
    const xs = Object.keys(sample.xs);
    const ys = Object.keys(sample.ys);

    xs.forEach(prop => {
      meta.inputs[prop] = {
        dtype: nnUtils.getDataType(sample.xs[prop]),
      };
    });

    ys.forEach(prop => {
      meta.outputs[prop] = {
        dtype: nnUtils.getDataType(sample.ys[prop]),
      };
    });

    // TODO: check if all entries have the same dtype.
    // otherwise throw an error

    this.meta = meta;

    return meta;
  }

  /**
   * ////////////////////////////////////////////////////////
   * Add Data
   * ////////////////////////////////////////////////////////
   */

  /**
   * Add Data
   * @param {object} xInputObj, {key: value}, key must be the name of the property value must be a String, Number, or Array
   * @param {*} yInputObj, {key: value}, key must be the name of the property value must be a String, Number, or Array
   */
  addData(xInputObj, yInputObj) {
    this.data.raw.push({
      xs: xInputObj,
      ys: yInputObj,
    });
  }

  /**
   * ////////////////////////////////////////////////////////
   * Tensor handling
   * ////////////////////////////////////////////////////////
   */

  /**
   * convertRawToTensors
   * converts array of {xs, ys} to tensors
   * @param {*} _dataRaw
   * @param {*} meta
   */
  // eslint-disable-next-line class-methods-use-this, no-unused-vars
  convertRawToTensors(dataRaw) {
    const meta = Object.assign({}, this.meta);
    const dataLength = dataRaw.length;

    return tf.tidy(() => {
      const inputArr = [];
      const outputArr = [];

      dataRaw.forEach(row => {
        // get xs
        const xs = Object.keys(meta.inputs)
          .map(k => {
            return row.xs[k];
          })
          .flat();

        inputArr.push(xs);

        // get ys
        const ys = Object.keys(meta.outputs)
          .map(k => {
            return row.ys[k];
          })
          .flat();

        outputArr.push(ys);
      });

      const inputs = tf.tensor(inputArr.flat(), [dataLength, ...meta.inputUnits]);
      const outputs = tf.tensor(outputArr.flat(), [dataLength, meta.outputUnits]);

      return {
        inputs,
        outputs,
      };
    });
  }

  /**
   * ////////////////////////////////////////////////////////
   * data normalization / unnormalization
   * ////////////////////////////////////////////////////////
   */

  /**
   * normalize the dataRaw input
   * @param {*} dataRaw
   */
  normalizeDataRaw(dataRaw) {
    const meta = Object.assign({}, this.meta);

    const normXs = this.normalizeInputData(dataRaw, meta.inputs, 'xs');
    const normYs = this.normalizeInputData(dataRaw, meta.outputs, 'ys');

    const normalizedData = nnUtils.zipArrays(normXs, normYs);

    return normalizedData;
  }

  /**
   * normalizeRaws
   * @param {*} dataRaw
   * @param {*} inputOrOutputMeta
   * @param {*} xsOrYs
   */
  // eslint-disable-next-line no-unused-vars, class-methods-use-this
  normalizeInputData(dataRaw, inputOrOutputMeta, xsOrYs) {
    // the data length
    const dataLength = dataRaw.length;
    // the copy of the inputs.meta[inputOrOutput]
    const inputMeta = Object.assign({}, inputOrOutputMeta);

    // normalized output object
    const normalized = {};
    Object.keys(inputMeta).forEach(k => {
      // get the min and max values
      const options = {
        min: inputMeta[k].min,
        max: inputMeta[k].max,
      };

      const dataAsArray = dataRaw.map(item => item[xsOrYs][k]);
      // depending on the input type, normalize accordingly
      if (inputMeta[k].dtype === 'string') {
        options.legend = inputMeta[k].legend;
        normalized[k] = this.normalizeArray(dataAsArray, options);
      } else if (inputMeta[k].dtype === 'number') {
        normalized[k] = this.normalizeArray(dataAsArray, options);
      } else if (inputMeta[k].dtype === 'array') {
        normalized[k] = dataAsArray.map(item => this.normalizeArray(item, options));
      }
    });

    // create a normalized version of data.raws
    const output = [...new Array(dataLength).fill(null)].map((item, idx) => {
      const row = {
        [xsOrYs]: {},
      };

      Object.keys(inputMeta).forEach(k => {
        row[xsOrYs][k] = normalized[k][idx];
      });

      return row;
    });

    return output;
  }

  /**
   * normalizeArray
   * @param {*} _input
   * @param {*} _options
   */
  // eslint-disable-next-line no-unused-vars, class-methods-use-this
  normalizeArray(inputArray, options) {
    const { min, max } = options;

    // if the data are onehot encoded, replace the string
    // value with the onehot array
    // if none exists, return the given value
    if (options.legend) {
      const normalized = inputArray.map(v => {
        return options.legend[v] ? options.legend[v] : v;
      });
      return normalized;
    }

    // if the dtype is a number
    if (inputArray.every(v => typeof v === 'number')) {
      const normalized = inputArray.map(v => nnUtils.normalizeValue(v, min, max));
      return normalized;
    }

    // otherwise return the input array
    // return inputArray;
    throw new Error('error in inputArray of normalizeArray() function');
  }

  /**
   * unNormalizeArray
   * @param {*} _input
   * @param {*} _options
   */
  // eslint-disable-next-line no-unused-vars, class-methods-use-this
  unnormalizeArray(inputArray, options) {
    const { min, max } = options;

    // if the data is onehot encoded then remap the
    // values from those oneHot arrays
    if (options.legend) {
      const unnormalized = inputArray.map(v => {
        let res;
        Object.entries(options.legend).forEach(item => {
          const key = item[0];
          const val = item[1];
          const matches = v.map((num, idx) => num === val[idx]).every(truthy => truthy === true);
          if (matches) res = key;
        });
        return res;
      });

      return unnormalized;
    }

    // if the dtype is a number
    if (inputArray.every(v => typeof v === 'number')) {
      const unnormalized = inputArray.map(v => nnUtils.unnormalizeValue(v, min, max));
      return unnormalized;
    }

    // otherwise return the input array
    // return inputArray;
    throw new Error('error in inputArray of normalizeArray() function');
  }

  /*
   * ////////////////////////////////////////////////
   * One hot encoding handling
   * ////////////////////////////////////////////////
   */

  /**
   * applyOneHotEncodingsToDataRaw
   * does not set this.data.raws
   * but rather returns them
   * @param {*} _dataRaw
   * @param {*} _meta
   */
  applyOneHotEncodingsToDataRaw(dataRaw) {
    const meta = Object.assign({}, this.meta);

    const output = dataRaw.map(row => {
      const xs = {
        ...row.xs,
      };
      const ys = {
        ...row.ys,
      };
      // get xs
      Object.keys(meta.inputs).forEach(k => {
        if (meta.inputs[k].legend) {
          xs[k] = meta.inputs[k].legend[row.xs[k]];
        }
      });

      Object.keys(meta.outputs).forEach(k => {
        if (meta.outputs[k].legend) {
          ys[k] = meta.outputs[k].legend[row.ys[k]];
        }
      });

      return {
        xs,
        ys,
      };
    });

    return output;
  }

  /**
   * getDataOneHot
   * creates onehot encodings for the input and outputs
   * and adds them to the meta info
   * @param {*} dataRaw
   */
  getDataOneHot(dataRaw) {
    const meta = Object.assign({}, this.meta);

    const inputMeta = this.getInputMetaOneHot(dataRaw, meta.inputs, 'xs');
    const outputMeta = this.getInputMetaOneHot(dataRaw, meta.outputs, 'ys');

    meta.inputs = inputMeta;
    meta.outputs = outputMeta;

    this.meta = {
      ...this.meta,
      ...meta,
    };

    return meta;
  }

  /**
   * getOneHotMeta
   * @param {*} _inputsMeta
   * @param {*} _dataRaw
   * @param {*} xsOrYs
   */
  getInputMetaOneHot(_dataRaw, _inputsMeta, xsOrYs) {
    const inputsMeta = Object.assign({}, _inputsMeta);

    Object.entries(inputsMeta).forEach(arr => {
      // the key
      const key = arr[0];
      // the value
      const { dtype } = arr[1];

      if (dtype === 'string') {
        const uniqueVals = [...new Set(_dataRaw.map(obj => obj[xsOrYs][key]))];
        const oneHotMeta = this.createOneHotEncodings(uniqueVals);
        inputsMeta[key] = {
          ...inputsMeta[key],
          ...oneHotMeta,
        };
      }
    });

    return inputsMeta;
  }

  /**
   * Returns a legend mapping the
   * data values to oneHot encoded values
   */
  // eslint-disable-next-line class-methods-use-this, no-unused-vars
  createOneHotEncodings(_uniqueValuesArray) {
    return tf.tidy(() => {
      const output = {
        uniqueValues: _uniqueValuesArray,
        legend: {},
      };

      const uniqueVals = _uniqueValuesArray; // [...new Set(this.data.raw.map(obj => obj.xs[prop]))]
      // get back values from 0 to the length of the uniqueVals array
      const onehotValues = uniqueVals.map((item, idx) => idx);
      // oneHot encode the values in the 1d tensor
      const oneHotEncodedValues = tf.oneHot(tf.tensor1d(onehotValues, 'int32'), uniqueVals.length);
      // convert them from tensors back out to an array
      const oneHotEncodedValuesArray = oneHotEncodedValues.arraySync();

      // populate the legend with the key/values
      uniqueVals.forEach((uVal, uIdx) => {
        output.legend[uVal] = oneHotEncodedValuesArray[uIdx];
      });

      return output;
    });
  }

  /**
   * ////////////////////////////////////////////////
   * saving / loading data
   * ////////////////////////////////////////////////
   */

  /**
   * Loads data from a URL using the appropriate function
   * @param {*} dataUrl
   * @param {*} inputs
   * @param {*} outputs
   */
  async loadDataFromUrl(dataUrl, inputs, outputs) {
    try {
      let result;

      if (dataUrl.endsWith('.csv')) {
        result = await this.loadCSV(dataUrl, inputs, outputs);
      } else if (dataUrl.endsWith('.json')) {
        result = await this.loadJSON(dataUrl, inputs, outputs);
      } else if (dataUrl.includes('blob')) {
        result = await this.loadBlob(dataUrl, inputs, outputs);
      } else {
        throw new Error('Not a valid data format. Must be csv or json');
      }

      return result;
    } catch (error) {
      console.error(error);
      throw new Error(error);
    }
  }

  /**
   * loadJSON
   * @param {*} _dataUrlOrJson
   * @param {*} _inputLabelsArray
   * @param {*} _outputLabelsArray
   */
  async loadJSON(dataUrlOrJson, inputLabels, outputLabels) {
    try {
      let json;
      // handle loading parsedJson
      if (dataUrlOrJson instanceof Object) {
        json = Object.assign({}, dataUrlOrJson);
      } else {
        const {data} = await axios.get(dataUrlOrJson);
        json = data;
      }

      // format the data.raw array
      const result = this.formatRawData(json, inputLabels, outputLabels);
      return result;
    } catch (err) {
      console.error('error loading json');
      throw new Error(err);
    }
  }

  /**
   * loadCSV
   * @param {*} _dataUrl
   * @param {*} _inputLabelsArray
   * @param {*} _outputLabelsArray
   */
  async loadCSV(dataUrl, inputLabels, outputLabels) {
    try {
      const myCsv = tf.data.csv(dataUrl);
      const loadedData = await myCsv.toArray();
      const json = {
        entries: loadedData,
      };
      // format the data.raw array
      const result = this.formatRawData(json, inputLabels, outputLabels);
      return result;
    } catch (err) {
      console.error('error loading csv', err);
      throw new Error(err);
    }
  }

  /**
   * loadBlob
   * @param {*} _dataUrlOrJson
   * @param {*} _inputLabelsArray
   * @param {*} _outputLabelsArray
   */
  async loadBlob(dataUrlOrJson, inputLabels, outputLabels) {
    try {
      const {data} = await axios.get(dataUrlOrJson);
      const text = data; // await data.text();

      let result;
      if (nnUtils.isJsonOrString(text)) {
        const json = JSON.parse(text);
        result = await this.loadJSON(json, inputLabels, outputLabels);
      } else {
        const json = this.csvToJSON(text);
        result = await this.loadJSON(json, inputLabels, outputLabels);
      }

      return result;
    } catch (err) {
      console.log('mmm might be passing in a string or something!', err);
      throw new Error(err);
    }
  }

  /**
   * loadData from fileinput or path
   * @param {*} filesOrPath
   * @param {*} callback
   */
  async loadData(filesOrPath = null, callback) {
    try {
      let loadedData;

      if (typeof filesOrPath !== 'string') {
        const file = filesOrPath[0];
        const fr = new FileReader();
        fr.readAsText(file);
        if (file.name.includes('.json')) {
          const temp = await file.text();
          loadedData = JSON.parse(temp);
        } else {
          console.log('data must be a json object containing an array called "data" or "entries');
        }
      } else {
        loadedData = await axios.get(filesOrPath, {responseType:"text"});
        const text = JSON.stringify(loadedData.data);
        if (nnUtils.isJsonOrString(text)) {
          loadedData = JSON.parse(text);
        } else {
          console.log(
            'Whoops! something went wrong. Either this kind of data is not supported yet or there is an issue with .loadData',
          );
        }
      }

      this.data.raw = this.findEntries(loadedData);

      // check if a data or entries property exists
      if (!this.data.raw.length > 0) {
        console.log('data must be a json object containing an array called "data" ');
      }

      if (callback) {
        callback();
      }
    } catch (error) {
      throw new Error(error);
    }
  }

  /**
   * saveData
   * @param {*} name
   */
  async saveData(name) {
    const today = new Date();
    const date = `${String(today.getFullYear())}-${String(today.getMonth() + 1)}-${String(
      today.getDate(),
    )}`;
    const time = `${String(today.getHours())}-${String(today.getMinutes())}-${String(
      today.getSeconds(),
    )}`;
    const datetime = `${date}_${time}`;

    let dataName = datetime;
    if (name) dataName = name;

    const output = {
      data: this.data.raw,
    };

    await saveBlob(JSON.stringify(output), `${dataName}.json`, 'text/plain');
  }

  /**
   * Saves metadata of the data
   * @param {*} nameOrCb
   * @param {*} cb
   */
  async saveMeta(nameOrCb, cb) {
    let modelName;
    let callback;

    if (typeof nameOrCb === 'function') {
      modelName = 'model';
      callback = nameOrCb;
    } else if (typeof nameOrCb === 'string') {
      modelName = nameOrCb;

      if (typeof cb === 'function') {
        callback = cb;
      }
    } else {
      modelName = 'model';
    }

    await saveBlob(JSON.stringify(this.meta), `${modelName}_meta.json`, 'text/plain');
    if (callback) {
      callback();
    }
  }

  /**
   * load a model and metadata
   * @param {*} filesOrPath
   * @param {*} callback
   */
  async loadMeta(filesOrPath = null, callback) {
    if (filesOrPath instanceof FileList) {
      const files = await Promise.all(
        Array.from(filesOrPath).map(async file => {
          if (file.name.includes('.json') && !file.name.includes('_meta')) {
            return {
              name: 'model',
              file,
            };
          } else if (file.name.includes('.json') && file.name.includes('_meta.json')) {
            const modelMetadata = await file.text();
            return {
              name: 'metadata',
              file: modelMetadata,
            };
          } else if (file.name.includes('.bin')) {
            return {
              name: 'weights',
              file,
            };
          }
          return {
            name: null,
            file: null,
          };
        }),
      );

      const modelMetadata = JSON.parse(files.find(item => item.name === 'metadata').file);

      this.meta = modelMetadata;
    } else if (filesOrPath instanceof Object) {
      // filesOrPath = {model: URL, metadata: URL, weights: URL}

      let modelMetadata = await axios.get(filesOrPath.metadata, {responseType:"text"});
      modelMetadata = JSON.stringify(modelMetadata.data);
      modelMetadata = JSON.parse(modelMetadata);

      this.meta = modelMetadata;
    } else {
      const metaPath = `${filesOrPath.substring(0, filesOrPath.lastIndexOf('/'))}/model_meta.json`;
      let modelMetadata = await axios.get(metaPath);
      modelMetadata = modelMetadata.data;

      this.meta = modelMetadata;
    }

    this.isMetadataReady = true;
    this.isWarmedUp = true;

    if (callback) {
      callback();
    }
    return this.meta;
  }

  /*
   * ////////////////////////////////////////////////
   * data loading helpers
   * ////////////////////////////////////////////////
   */

  /**
   * // TODO: convert ys into strings, if the task is classification
    // if (this.config.architecture.task === "classification" && typeof output.ys[prop] !== "string") {
    //   output.ys[prop] += "";
    // }
   * formatRawData
   * takes a json and set the this.data.raw
   * @param {*} json 
   * @param {Array} inputLabels
   * @param {Array} outputLabels
   */
  formatRawData(json, inputLabels, outputLabels) {
    // Recurse through the json object to find
    // an array containing `entries` or `data`
    const dataArray = this.findEntries(json);

    if (!dataArray.length > 0) {
      console.log(`your data must be contained in an array in \n
        a property called 'entries' or 'data' of your json object`);
    }

    // create an array of json objects [{xs,ys}]
    const result = dataArray.map((item, idx) => {
      const output = {
        xs: {},
        ys: {},
      };

      inputLabels.forEach(k => {
        if (item[k] !== undefined) {
          output.xs[k] = item[k];
        } else {
          console.error(`the input label ${k} does not exist at row ${idx}`);
        }
      });

      outputLabels.forEach(k => {
        if (item[k] !== undefined) {
          output.ys[k] = item[k];
        } else {
          console.error(`the output label ${k} does not exist at row ${idx}`);
        }
      });

      return output;
    });

    // set this.data.raw
    this.data.raw = result;

    return result;
  }

  /**
   * csvToJSON
   * Creates a csv from a string
   * @param {*} csv
   */
  // via: http://techslides.com/convert-csv-to-json-in-javascript
  // eslint-disable-next-line class-methods-use-this
  csvToJSON(csv) {
    // split the string by linebreak
    const lines = csv.split('\n');
    const result = [];
    // get the header row as an array
    const headers = lines[0].split(',');

    // iterate through every row
    for (let i = 1; i < lines.length; i += 1) {
      // create a json object for each row
      const row = {};
      // split the current line into an array
      const currentline = lines[i].split(',');

      // for each header, create a key/value pair
      headers.forEach((k, idx) => {
        row[k] = currentline[idx];
      });
      // add this to the result array
      result.push(row);
    }

    return {
      entries: result,
    };
  }

  /**
   * findEntries
   * recursively attempt to find the entries
   * or data array for the given json object
   * @param {*} _data
   */
  findEntries(_data) {
    const parentCopy = Object.assign({}, _data);

    if (parentCopy.entries && parentCopy.entries instanceof Array) {
      return parentCopy.entries;
    } else if (parentCopy.data && parentCopy.data instanceof Array) {
      return parentCopy.data;
    }

    const keys = Object.keys(parentCopy);
    // eslint-disable-next-line consistent-return
    keys.forEach(k => {
      if (typeof parentCopy[k] === 'object') {
        return this.findEntries(parentCopy[k]);
      }
    });

    return parentCopy;
  }
  
  /**
   * getData
   * return data object's raw array
   * to make getting raw data easier
   */
  getData() {
    const rawArray = this.data.raw;
    return rawArray;
  }
}

export default NeuralNetworkData;
