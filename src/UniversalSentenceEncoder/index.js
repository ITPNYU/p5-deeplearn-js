// import * as tf from '@tensorflow/tfjs';
import * as USE from '@tensorflow-models/universal-sentence-encoder';
import callCallback from '../utils/callcallback';


/**
 * @typedef {Object} USEOptions
 * @property {boolean} withTokenizer
 */
const DEFAULTS = {
  withTokenizer: false,
}

/**
 * @property {(USE.UniversalSentenceEncoder | null)} model
 * @property {(USE.Tokenizer | null)} tokenizer
 * @property {USEOptions} config
 */
class UniversalSentenceEncoder {
  /**
   * @param {USEOptions} [options]
   * @param {function} callback
   */
  constructor(options, callback){
    this.model = null;
    this.tokenizer = null;
    this.config = {
      withTokenizer: options.withTokenizer || DEFAULTS.withTokenizer
    };

    callCallback(this.loadModel(), callback);
  }

  /**
   * @private
   * load model
   */
  async loadModel(){
    if(this.config.withTokenizer === true){
      this.tokenizer = await USE.loadTokenizer();
    } 
    this.model = await USE.load();
    return this;
  }

  /**
   * @public
   * @param {(string | string[])} text
   * @param {function} callback
   * @return {Promise}
   */
  predict(text, callback){
    return callCallback(this.predictInternal(text), callback);
  }

  /**
   * @private
   * @param {(string | string[])} text
   * @return {Promise<number[][]>}
   */
  async predictInternal(text){
    try {
      const embeddings = await this.model.embed(text);
      const results = await embeddings.array();
      embeddings.dispose();
      return results;
    }
    // TODO: it would be better not to catch the error
    catch(err) {
      console.error(err);
      return err;
    }
  }

  /**
   * Encodes a string based on the loaded tokenizer if the withTokenizer:true
   * @public
   * @param {string} textString
   * @param {function} callback
   * @return {Promise}
   */
  encode(textString, callback){
    return callCallback(this.encodeInternal(textString), callback);
  }

  /**
   * @private
   * @param {string} textString
   * @return {Promise<boolean|Uint8Array>}
   */
  async encodeInternal(textString){
    if(this.config.withTokenizer === true){
      return this.tokenizer.encode(textString);
    }
    console.error('withTokenizer must be set to true - please pass "withTokenizer:true" as an option in the constructor');
    return false;
  }

}

/**
 * @param {function | Partial<USEOptions>} [optionsOr]
 * @param {function} [cb]
 * @return {UniversalSentenceEncoder}
 * // TODO: callCallback
 */
const universalSentenceEncoder = (optionsOr, cb) => {
  const options = (typeof optionsOr === 'object') ? optionsOr : {};
  const callback = (typeof optionsOr === 'function') ? optionsOr : cb;

  return new UniversalSentenceEncoder(options, callback);
};

export default universalSentenceEncoder;