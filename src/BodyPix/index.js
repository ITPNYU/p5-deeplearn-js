// Copyright (c) 2019 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/* eslint prefer-destructuring: ["error", {AssignmentExpression: {array: false}}] */
/* eslint no-await-in-loop: "off" */

/*
 * BodyPix: Real-time Person Segmentation in the Browser
 * Ported and integrated from all the hard work by: https://github.com/tensorflow/tfjs-models/tree/master/body-pix
 */

import * as tf from '@tensorflow/tfjs';
import * as bp from '@tensorflow-models/body-pix';
import callCallback from '../utils/callcallback';
import p5Utils from '../utils/p5Utils';
import BODYPIX_PALETTE from './BODYPIX_PALETTE';


const DEFAULTS = {
    multiplier: 0.5, // 1.0, 0.75, or 0.50
    flipHorizontal: false, // true
    outputStride: 16, // 8, 16, 32
    segmentationThreshold: 0.7, // 0 - 1
    palette: BODYPIX_PALETTE,
    internalResolution: 'medium', // low, medium, high, full or 0 - 1 
    quantBytes: 2, // 4, 2, 1
    nmsRadius: 10, 
    maxDetections: 10,
    architecture: 'MobileNetV1', // 'ResNet50'
    modelUrl: null,
    returnTensors: false // true
}

class BodyPix {
    /**
     * Create BodyPix.
     * @param {HTMLVideoElement} video - An HTMLVideoElement.
     * @param {object} options - An object with options.
     * @param {function} callback - A callback to be called when the model is ready.
     */
    constructor(video, options, callback) {
        this.video = video;
        this.model = null;
        this.modelReady = false;
        this.config = {
            multiplier: options.multiplier || DEFAULTS.multiplier,
            flipHorizontal: options.flipHorizontal || DEFAULTS.flipHorizontal,
            outputStride: options.outputStride || DEFAULTS.outputStride,
            segmentationThreshold: options.segmentationThreshold || DEFAULTS.segmentationThreshold,
            palette: options.palette || DEFAULTS.palette,
            internalResolution: options.internalResolution || DEFAULTS.internalResolution,
            quantBytes: options.quantBytes || DEFAULTS.quantBytes,
            nmsRadius: options.nmsRadius || DEFAULTS.nmsRadius,
            maxDetections: options.maxDetections || DEFAULTS.maxDetections,
            architecture: options.architecture || DEFAULTS.architecture,
            modelUrl: options.modelUrl || DEFAULTS.modelUrl,
            returnTensors: options.returnTensors || DEFAULTS.returnTensors
        }

        this.ready = callCallback(this.loadModel(), callback);
    }

    /**
     * Load the model and set it to this.model
     * @return {this} the BodyPix model.
     */
    async loadModel() {
        const options = {
            architecture: this.config.architecture,
            outputStride: this.config.outputStride,
            multiplier: this.config.multiplier,
            quantBytes: this.config.quantBytes
        };
        
        if(this.config.modelUrl !== null){
            options.modelUrl = this.config.modelUrl;
        }

        this.model = await bp.load(options);
        this.modelReady = true;
        return this;
    }

    /**
     * Returns an rgb array
     * @param {Object} a p5.Color obj
     * @return {Array} an [r,g,b] array
     */
    /* eslint class-methods-use-this: "off" */
    p5Color2RGB(p5ColorObj) {
        const regExp = /\(([^)]+)\)/;
        const match = regExp.exec(p5ColorObj.toString('rgb'));
        const [r, g, b] = match[1].split(',')
        return [r, g, b]
    }

    /**
     * Returns a p5Image
     * @param {*} tfBrowserPixelImage 
     */
    async convertToP5Image(tfBrowserPixelImage, segmentationWidth, segmentationHeight) {
        const blob1 = await p5Utils.rawToBlob(tfBrowserPixelImage, segmentationWidth, segmentationHeight);
        const p5Image1 = await p5Utils.blobToP5Image(blob1);
        return p5Image1
    }

    /**
     * Returns a bodyPartsSpec object 
     * @param {Array} an array of [r,g,b] colors
     * @return {object} an object with the bodyParts by color and id
     */
    /* eslint class-methods-use-this: "off" */
    bodyPartsSpec(colorOptions) {
        const result = colorOptions !== undefined || Object.keys(colorOptions).length >= 24 ? colorOptions : this.config.palette;

        // Check if we're getting p5 colors, make sure they are rgb 
        if (p5Utils.checkP5() && result !== undefined && Object.keys(result).length >= 24) {
            // Ensure the p5Color object is an RGB array
            Object.keys(result).forEach(part => {
                if (result[part].color instanceof window.p5.Color) {
                    result[part].color = this.p5Color2RGB(result[part].color);
                } else {
                    result[part].color = result[part].color;
                }
            });
        }

        return result;
    }

    /**
     * Segments the image with partSegmentation, return result object
     * @param {HTMLImageElement | HTMLCanvasElement | object | function | number} imageToSegment - 
     *    takes any of the following params
     * @param {object} segmentationOptions - config params for the segmentation
     *    includes outputStride, segmentationThreshold
     * @return {Object} a result object with image, raw, bodyParts
     */
    async segmentWithPartsInternal(imgToSegment, segmentationOptions) {
        // estimatePartSegmentation
        await this.ready;
        await tf.nextFrame();

        if (this.video && this.video.readyState === 0) {
            await new Promise(resolve => {
                this.video.onloadeddata = () => resolve();
            });
        }

        this.config.palette = segmentationOptions.palette || this.config.palette;
        this.config.flipHorizontal = segmentationOptions.flipHorizontal || this.config.flipHorizontal;
        this.config.internalResolution = segmentationOptions.internalResolution || this.config.internalResolution;
        this.config.segmentationThreshold = segmentationOptions.segmentationThreshold || this.config.segmentationThreshold;
        this.config.maxDetections = segmentationOptions.maxDetections || this.config.maxDetections;
        this.config.scoreThreshold = segmentationOptions.scoreThreshold || this.config.scoreThreshold;
        this.config.nmsRadius = segmentationOptions.nmsRadius || this.config.nmsRadius;

        const segmentation = await this.model.segmentPersonParts(imgToSegment, {
            flipHorizontal: this.config.flipHorizontal,
            internalResolution: this.config.internalResolution,
            segmentationThreshold: this.config.segmentationThreshold,
            maxDetections: this.config.maxDetections,
            scoreThreshold: this.config.scoreThreshold,
            nmsRadius: this.config.nmsRadius
        });

        const bodyPartsMeta = this.bodyPartsSpec(this.config.palette);
        const colorsArray = Object.keys(bodyPartsMeta).map(part => bodyPartsMeta[part].color)

        const result = {
            segmentation,
            raw: {
                personMask: null,
                backgroundMask: null,
                partMask: null
            },
            tensor: {
                personMask: null,
                backgroundMask: null,
                partMask: null,
            },
            personMask: null,
            backgroundMask: null,
            partMask: null,
            bodyParts: bodyPartsMeta
        };
        result.raw.backgroundMask = bp.toMaskImageData(segmentation, true);
        result.raw.personMask = bp.toMaskImageData(segmentation, false);
        result.raw.partMask = bp.toColoredPartImageData(segmentation, colorsArray);

        const {
            personMask,
            backgroundMask,
            partMask,
        } = tf.tidy(() => {
            let normTensor = tf.browser.fromPixels(imgToSegment);
            // create a tensor from the input image
            const alpha = tf.ones([segmentation.height, segmentation.width, 1]).tile([1, 1, 1]).mul(255)
            normTensor = normTensor.concat(alpha, 2)

            // create a tensor from the segmentation
            let maskPersonTensor = tf.tensor(segmentation.data, [segmentation.height, segmentation.width, 1]);
            let maskBackgroundTensor = tf.tensor(segmentation.data, [segmentation.height, segmentation.width, 1]);
            let partTensor = tf.tensor([...result.raw.partMask.data], [segmentation.height, segmentation.width, 4]);

            // multiply the segmentation and the inputImage
            maskPersonTensor = tf.cast(maskPersonTensor.add(0.2).sign().relu().mul(normTensor), 'int32')
            maskBackgroundTensor = tf.cast(maskBackgroundTensor.add(0.2).sign().neg().relu().mul(normTensor), 'int32')
            // TODO: handle removing background 
            partTensor = tf.cast(partTensor, 'int32')

            return {
                personMask: maskPersonTensor,
                backgroundMask: maskBackgroundTensor,
                partMask: partTensor
            }
        })

        const personMaskPixels = await tf.browser.toPixels(personMask);
        const bgMaskPixels = await tf.browser.toPixels(backgroundMask);
        const partMaskPixels = await tf.browser.toPixels(partMask);

        // otherwise, return the pixels 
        result.personMask = personMaskPixels;
        result.backgroundMask = bgMaskPixels;
        result.partMask = partMaskPixels;

        // if p5 exists, convert to p5 image
        if (p5Utils.checkP5()) {
            result.personMask = await this.convertToP5Image(personMaskPixels, segmentation.width, segmentation.height)
            result.backgroundMask = await this.convertToP5Image(bgMaskPixels, segmentation.width, segmentation.height)
            result.partMask = await this.convertToP5Image(partMaskPixels, segmentation.width, segmentation.height)
        }

        if (!this.config.returnTensors) {
            personMask.dispose();
            backgroundMask.dispose();
            partMask.dispose();
        } else {
            // return tensors
            result.tensor.personMask = personMask;
            result.tensor.backgroundMask = backgroundMask;
            result.tensor.partMask = partMask;
        }

        return result;

    }

    /**
     * Segments the image with partSegmentation
     * @param {HTMLImageElement | HTMLCanvasElement | object | function | number} optionsOrCallback - 
     *    takes any of the following params
     * @param {object} configOrCallback - config params for the segmentation
     *    includes palette, outputStride, segmentationThreshold
     * @param {function} cb - a callback function that handles the results of the function.
     * @return {function} a promise or the results of a given callback, cb.
     */
    async segmentWithParts(optionsOrCallback, configOrCallback, cb) {
        let imgToSegment = this.video;
        let callback;
        let segmentationOptions = this.config;

        // Handle the image to predict
        if (typeof optionsOrCallback === 'function') {
            imgToSegment = this.video;
            callback = optionsOrCallback;
            // clean the following conditional statement up!
        } else if (optionsOrCallback instanceof HTMLImageElement ||
            optionsOrCallback instanceof HTMLCanvasElement ||
            optionsOrCallback instanceof HTMLVideoElement ||
            optionsOrCallback instanceof ImageData) {
            imgToSegment = optionsOrCallback;
        } else if (typeof optionsOrCallback === 'object' && (optionsOrCallback.elt instanceof HTMLImageElement ||
                optionsOrCallback.elt instanceof HTMLCanvasElement ||
                optionsOrCallback.elt instanceof ImageData)) {
            imgToSegment = optionsOrCallback.elt; // Handle p5.js image
        } else if (typeof optionsOrCallback === 'object' && optionsOrCallback.canvas instanceof HTMLCanvasElement) {
            imgToSegment = optionsOrCallback.canvas; // Handle p5.js image
        } else if (typeof optionsOrCallback === 'object' && optionsOrCallback.elt instanceof HTMLVideoElement) {
            imgToSegment = optionsOrCallback.elt; // Handle p5.js image
        } else if (!(this.video instanceof HTMLVideoElement)) {
            // Handle unsupported input
            throw new Error(
                'No input image provided. If you want to classify a video, pass the video element in the constructor. ',
            );
        }

        if (typeof configOrCallback === 'object') {
            segmentationOptions = configOrCallback;
        } else if (typeof configOrCallback === 'function') {
            callback = configOrCallback;
        }

        if (typeof cb === 'function') {
            callback = cb;
        }

        return callCallback(this.segmentWithPartsInternal(imgToSegment, segmentationOptions), callback);

    }

    /**
     * Segments the image with personSegmentation, return result object
     * @param {HTMLImageElement | HTMLCanvasElement | object | function | number} imageToSegment - 
     *    takes any of the following params
     * @param {object} segmentationOptions - config params for the segmentation
     *    includes outputStride, segmentationThreshold
     * @return {Object} a result object with maskBackground, maskPerson, raw
     */
    async segmentInternal(imgToSegment, segmentationOptions) {

        await this.ready;
        await tf.nextFrame();

        if (this.video && this.video.readyState === 0) {
            await new Promise(resolve => {
                this.video.onloadeddata = () => resolve();
            });
        }

        // this.config.flipHorizontal = segmentationOptions.flipHorizontal || this.config.flipHorizontal;
        // this.config.internalResolution = segmentationOptions.internalResolution || this.config.internalResolution;
        // this.config.segmentationThreshold = segmentationOptions.segmentationThreshold || this.config.segmentationThreshold;
        // this.config.maxDetections = segmentationOptions.maxDetections || this.config.maxDetections;
        // this.config.scoreThreshold = segmentationOptions.scoreThreshold || this.config.scoreThreshold;
        // this.config.nmsRadius = segmentationOptions.nmsRadius || this.config.nmsRadius;

        const segmentation = await this.model.segmentPerson(imgToSegment, {
            flipHorizontal: this.config.flipHorizontal,
            internalResolution: this.config.internalResolution,
            segmentationThreshold: this.config.segmentationThreshold,
            maxDetections: this.config.maxDetections,
            scoreThreshold: this.config.scoreThreshold,
            nmsRadius: this.config.nmsRadius,
            ...segmentationOptions
        });

        const result = {
            segmentation,
            raw: {
                personMask: null,
                backgroundMask: null,
            },
            tensor: {
                personMask: null,
                backgroundMask: null,
            },
            personMask: null,
            backgroundMask: null,
        };
        result.raw.personMask = bp.toMask(segmentation, {r: 255, g: 255, b: 255, a: 255}, {r: 0, g: 0, b: 0, a: 255});
        result.raw.backgroundMask = bp.toMask(segmentation, {r: 255, g: 255, b: 255, a: 255}, {r: 0, g: 0, b: 0, a: 255});
        

        // TODO: consider returning the canvas with the bp.drawMask()
        // const bgMaskCanvas = document.createElement('canvas');
        // bgMaskCanvas.width = segmentation.width;
        // bgMaskCanvas.height = segmentation.height;
        // bp.drawMask(bgMaskCanvas, imgToSegment, result.maskBackground, 1, 3, false);

        // const featureMaskCanvas = document.createElement('canvas');
        // featureMaskCanvas.width = segmentation.width;
        // featureMaskCanvas.height = segmentation.height;
        // bp.drawMask(featureMaskCanvas, imgToSegment, result.maskPerson, 1, 3, false);

        // result.backgroundMask = bgMaskCanvas;
        // result.featureMask = featureMaskCanvas;

        const {
            personMask,
            backgroundMask
        } = tf.tidy(() => {
            let normTensor = tf.browser.fromPixels(imgToSegment);
            // create a tensor from the input image
            const alpha = tf.ones([segmentation.height, segmentation.width, 1]).tile([1, 1, 1]).mul(255)
            normTensor = normTensor.concat(alpha, 2)
            // normTensor.print();

            // create a tensor from the segmentation
            let maskPersonTensor = tf.tensor(segmentation.data, [segmentation.height, segmentation.width, 1]);
            let maskBackgroundTensor = tf.tensor(segmentation.data, [segmentation.height, segmentation.width, 1]);

            // multiply the segmentation and the inputImage
            maskPersonTensor = tf.cast(maskPersonTensor.neg().add(1).mul(normTensor), 'int32')
            maskBackgroundTensor = tf.cast(maskBackgroundTensor.mul(normTensor), 'int32')

            return {
                personMask: maskPersonTensor,
                backgroundMask: maskBackgroundTensor,
            }
        })

        const personMaskPixels = await tf.browser.toPixels(personMask);
        const bgMaskPixels = await tf.browser.toPixels(backgroundMask);

        // // if p5 exists, convert to p5 image
        if (p5Utils.checkP5()) {
            result.personMask = await this.convertToP5Image(personMaskPixels, segmentation.width, segmentation.height)
            result.backgroundMask = await this.convertToP5Image(bgMaskPixels, segmentation.width, segmentation.height)
        } else {
            // otherwise, return the pixels 
            result.personMask = personMaskPixels;
            result.backgroundMask = bgMaskPixels;
        }

        if (!this.config.returnTensors) {
            personMask.dispose();
            backgroundMask.dispose();
        } else {
            result.tensor.personMask = personMask;
            result.tensor.backgroundMask = backgroundMask;
        }

        return result;

    }

    /**
     * Segments the image with personSegmentation
     * @param {HTMLImageElement | HTMLCanvasElement | object | function | number} optionsOrCallback - 
     *    takes any of the following params
     * @param {object} configOrCallback - config params for the segmentation
     *    includes outputStride, segmentationThreshold
     * @param {function} cb - a callback function that handles the results of the function.
     * @return {function} a promise or the results of a given callback, cb.
     */
    async segment(optionsOrCallback, configOrCallback, cb) {
        let imgToSegment = this.video;
        let callback;
        let segmentationOptions = this.config;

        // Handle the image to predict
        if (typeof optionsOrCallback === 'function') {
            imgToSegment = this.video;
            callback = optionsOrCallback;
            // clean the following conditional statement up!
        } else if (optionsOrCallback instanceof HTMLImageElement ||
            optionsOrCallback instanceof HTMLCanvasElement ||
            optionsOrCallback instanceof HTMLVideoElement ||
            optionsOrCallback instanceof ImageData) {
            imgToSegment = optionsOrCallback;
        } else if (typeof optionsOrCallback === 'object' && (optionsOrCallback.elt instanceof HTMLImageElement ||
                optionsOrCallback.elt instanceof HTMLCanvasElement ||
                optionsOrCallback.elt instanceof ImageData)) {
            imgToSegment = optionsOrCallback.elt; // Handle p5.js image
        } else if (typeof optionsOrCallback === 'object' && optionsOrCallback.canvas instanceof HTMLCanvasElement) {
            imgToSegment = optionsOrCallback.canvas; // Handle p5.js image
        } else if (typeof optionsOrCallback === 'object' && optionsOrCallback.elt instanceof HTMLVideoElement) {
            imgToSegment = optionsOrCallback.elt; // Handle p5.js image
        } else if (!(this.video instanceof HTMLVideoElement)) {
            // Handle unsupported input
            throw new Error(
                'No input image provided. If you want to classify a video, pass the video element in the constructor. ',
            );
        }

        if (typeof configOrCallback === 'object') {
            segmentationOptions = configOrCallback;
        } else if (typeof configOrCallback === 'function') {
            callback = configOrCallback;
        }

        if (typeof cb === 'function') {
            callback = cb;
        }

        return callCallback(this.segmentInternal(imgToSegment, segmentationOptions), callback);
    }

}

const bodyPix = (videoOrOptionsOrCallback, optionsOrCallback, cb) => {
    let video;
    let options = {};
    let callback = cb;

    if (videoOrOptionsOrCallback instanceof HTMLVideoElement) {
        video = videoOrOptionsOrCallback;
    } else if (
        typeof videoOrOptionsOrCallback === 'object' &&
        videoOrOptionsOrCallback.elt instanceof HTMLVideoElement
    ) {
        video = videoOrOptionsOrCallback.elt; // Handle a p5.js video element
    } else if (typeof videoOrOptionsOrCallback === 'object') {
        options = videoOrOptionsOrCallback;
    } else if (typeof videoOrOptionsOrCallback === 'function') {
        callback = videoOrOptionsOrCallback;
    }

    if (typeof optionsOrCallback === 'object') {
        options = optionsOrCallback;
    } else if (typeof optionsOrCallback === 'function') {
        callback = optionsOrCallback;
    }

    const instance = new BodyPix(video, options, callback);
    return callback ? instance : instance.ready;
}

export default bodyPix;