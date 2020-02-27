/* eslint-disable class-methods-use-this */
// Copyright (c) 2018 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT
/* eslint max-len: ["error", { "code": 180 }] */

/*
YOLO Object detection
Heavily derived from https://github.com/ModelDepot/tfjs-yolo-tiny (ModelDepot: modeldepot.io)
AND https://github.com/TheHidden1/yolo.js
*/

import * as tf from '@tensorflow/tfjs';
import Video from './../../utils/Video';
import { imgToTensorYOLO, isInstanceOfSupportedElement } from './../../utils/imageUtilities';

import isTensorArray from './../../utils/tensorUtilities';
import callCallback from './../../utils/callcallback';
import CLASS_NAMES from './../../utils/COCO_CLASSES';
import modelLoader from './../../utils/modelLoader';


// these are the default you yolo v2
const DEFAULTS = {
	modelName: 'tiny-yolo-v2',
	modelUrl: 'https://raw.githubusercontent.com/ml5js/ml5-data-and-training/master/models/YOLO/model.json',
	filterBoxesThreshold: 0.01,
	IOUThreshold: 0.4,
	classProbThreshold: 0.4,
	modelSize: [224, 224],
	maxOutput: 10,
	masks: [[0, 1, 2, 3, 4]],
	anchors: [
		[0.57273, 0.677385],
		[1.87446, 2.06253],
		[3.33843, 5.47434],
		[7.88282, 3.52778],
		[9.77052, 9.16828],
	],
};

class YOLOBase extends Video {

	/**
	 * @deprecated Please use ObjectDetector class instead
	 */
	/**
	 * @typedef {Object} options
	 * @property {number} filterBoxesThreshold - default 0.01
	 * @property {number} IOUThreshold - default 0.4
	 * @property {number} classProbThreshold - default 0.4
	 */
	/**
	 * Create YOLO model. Works on video and images.
	 * @param {HTMLVideoElement|HTMLImageElement|HTMLCanvasElement|ImageData} video - Optional. The video to be used for object detection and classification.
	 * @param {Object} options - Optional. A set of options.
	 * @param {function} callback - Optional. A callback function that is called once the model has loaded.
	 */
	constructor(video, options, callback) {
		const imageSize = options.modelSize[0] || DEFAULTS.modelSize[0]
		super(video, imageSize);

		this.modelUrl = options.modelUrl || DEFAULTS.modelUrl;
		this.modelName = options.modelName || DEFAULTS.modelName;


		this.modelSize = options.modelSize || DEFAULTS.modelSize;
		this.filterBoxesThreshold = options.filterBoxesThreshold || DEFAULTS.filterBoxesThreshold;
		this.IOUThreshold = options.IOUThreshold || DEFAULTS.IOUThreshold;
		this.classProbThreshold = options.classProbThreshold || DEFAULTS.classProbThreshold;
		this.modelSize = options.modelSize || DEFAULTS.modelSize;
		this.maxOutput = options.maxOutput || DEFAULTS.maxOutput;
		this.masks = options.masks || DEFAULTS.masks;
		this.anchors = options.anchors || DEFAULTS.anchors;



		this.modelReady = false;
		this.isPredicting = false;
		this.callback = callback;
		this.ready = callCallback(this.loadModel(), this.callback);

		if (!options.disableDeprecationNotice) {
			console.warn(
				'WARNING! Function YOLO has been deprecated, please use the new ObjectDetector function instead',
			);
		}
	}

	async loadModel() {
		if (this.videoElt && !this.video) {
			this.video = await this.loadVideo();
		}

		if (modelLoader.isAbsoluteURL(this.modelUrl) === true) {
			this.model = await tf.loadLayersModel(this.modelUrl);
		} else {
			const modelPath = modelLoader.getModelPath(this.modelUrl);
			this.modelUrl = `${modelPath}/model.json`;
			this.model = await tf.loadLayersModel(this.modelUrl);
		}

		this.modelReady = true;
		return this;
	}

	/**
	 * Detect objects that are in video, returns bounding box, label, and confidence scores
	 * @param {HTMLVideoElement|HTMLImageElement|HTMLCanvasElement|ImageData} subject - Subject of the detection.
	 * @param {function} callback - Optional. A callback function that is called once the model has loaded. If no callback is provided, it will return a promise
	 *    that will be resolved once the prediction is done.
	 * @returns {ObjectDetectorPrediction}
	 */
	async detect(inputOrCallback, cb) {
		await this.ready;
		let imgToPredict;
		let callback = cb;

		if (isInstanceOfSupportedElement(inputOrCallback)) {
			imgToPredict = inputOrCallback;
		} else if (
			typeof inputOrCallback === 'object' &&
			isInstanceOfSupportedElement(inputOrCallback.elt)
		) {
			imgToPredict = inputOrCallback.elt; // Handle p5.js image and video.
		} else if (
			typeof inputOrCallback === 'object' &&
			isInstanceOfSupportedElement(inputOrCallback.canvas)
		) {
			imgToPredict = inputOrCallback.canvas; // Handle p5.js image and video.
		} else if (typeof inputOrCallback === 'function') {
			imgToPredict = this.video;
			callback = inputOrCallback;
		} else {
			throw new Error('Detection subject not supported');
		}

		return callCallback(this.detectInternal(imgToPredict), callback);
	}

	/**
	 * Detect objects that are in video, returns bounding box, label, and confidence scores
	 * @param {HTMLVideoElement|HTMLImageElement|HTMLCanvasElement|ImageData} subject - Subject of the detection.
	 * @returns {ObjectDetectorPrediction}
	 */
	async detectInternal(imgToPredict) {
		await this.ready;
		await tf.nextFrame();

		this.isPredicting = true;
		let originalImageSize = null;
		const [rawBoxes, rawScores] = tf.tidy(() => {
			const [ originalImageShape , input ] = imgToTensorYOLO(imgToPredict, this.modelSize);
			originalImageSize = originalImageShape;
			const activation = this.model.predict(input);			
			const [rawboxes, rawscores] = this.postProcessRawPredictions(activation);

			// removes batch dim
			const boxes = rawboxes.squeeze([0]);
			const scores = rawscores.squeeze([0]);
			return [boxes, scores];
		});
		const [boxes, scores, classes] = await this.postProcessBoxes(rawBoxes, rawScores);

		const dectections = this.createDetectionArray(boxes,scores,classes, this.modelSize, originalImageSize);
		this.isPredicting = false;
		return dectections;
	}

	/**
	 * the postprocessing function for the yolo object detection algorithm
	 * @param rawPrediction can be a `tf.Tensor` representing a single output (yolov2)
	 * or a `tf.Tensor[]` representing multiple outputs (yolov3 has 3 outputs  at 3 diffrent scales).
	 * each output has the shape of `[batch, size, size, ( numClasses + 5 ) * numAnchors]`
	 * with the 5 representing: Box Coodinates [4] + BoxConfidence [1]
	 *
	 * @return a `tf.Tensor[]` that contain `[Boxes, Scores]`
	 * `Boxes` with a shape of `[batch, numBoxes, 4]`
	 *  Each `box` is defined by `[topY, topX, bottomY, bottomX]`
	 *
	 * `Scores` with a shape of `[batch, numBoxes, numClasses]`
	 */
	postProcessRawPredictions(rawPrediction) {
		const layers = [];
		let isV3 = false;

		if (isTensorArray(rawPrediction)) {
			for (let i = 0; i < rawPrediction.length; i += 1) {
				layers.push(rawPrediction[i]);
			}
			isV3 = true;
		} else {
			layers.push(rawPrediction);
		}

		const boxes = [];
		const probs = [];

		for (let i = 0; i < layers.length; i += 1) {
			const mask = this.masks[i];
			const anchors = tf.gather(this.anchors, mask).expandDims(0);
			const [box, prob] = this.processLayer(layers[i], anchors, this.modelSize, isV3);
			boxes.push(box);
			probs.push(prob);
		}

		const boxesTensor = tf.concat(boxes, 1);
		const probsTensor = tf.concat(probs, 1);

		return [boxesTensor, probsTensor];
	}

	/**
	 * Process 1 layer of the yolo output
	 * @param prediction a `tf.Tensor` representing 1 output of  the neural net
	 * @param anchorsTensor a `tf.Tensor` representing the anchors that correspond with the output
	 * shape: `[numAnchors, 2]`
	 * @param modelSize the input size for the neural net
	 * @param isV3 boolean indecating if its a v2 or a v3 model
	 *
	 * @return a `tf.Tensor[]` that containes `[boxes , Scores]` that correspond to the specific layer
	 */
	processLayer(prediction, anchors, modelSize, isV3) {

		const numAnchors = anchors.shape[1];
		const [batch, outputHeight, outputWidth, data] = prediction.shape;
		const numBoxes = outputWidth * outputHeight * numAnchors;
		const dataTensorLen = data / numAnchors;
		const numClasses = dataTensorLen - 5;

		const reshaped = tf.reshape(prediction, [batch, outputHeight, outputWidth, numAnchors, dataTensorLen]);

		const scalefactor = [modelSize[0] / outputHeight, modelSize[1] / outputWidth];

		let [boxXY, boxWH, boxConfidence, boxClassProbs] = tf.split(reshaped, [2, 2, 1, numClasses], 4);

		const gridX = tf.tile(tf.reshape(tf.range(0, outputWidth), [1, -1, 1, 1]), [outputHeight, 1, 1, 1]);
		const gridY = tf.tile(tf.reshape(tf.range(0, outputHeight), [-1, 1, 1, 1]), [1, outputWidth, 1, 1]);
		
		let indexGrid = tf.concat([gridX, gridY], 3);
		indexGrid = tf.tile(indexGrid, [1, 1, numAnchors, 1]);

		let anchorsTensor = anchors;
		if (isV3) {
			anchorsTensor = anchors.div(scalefactor);
		}

		boxXY = tf.div(tf.add(tf.sigmoid(boxXY), indexGrid), [outputWidth, outputHeight]);
		boxWH = tf.div(tf.mul(tf.exp(boxWH), anchorsTensor), [outputWidth, outputHeight]);

		const boxYX = tf.concat(tf.split(boxXY, 2, 4).reverse(), 4);
		const boxHW = tf.concat(tf.split(boxWH, 2, 4).reverse(), 4);

		// XY WH to XmaxYmax XminYmin
		const boxMins = tf.mul(tf.sub(boxYX, tf.div(boxHW, 2)), modelSize);
		const boxMaxes = tf.mul(tf.add(boxYX, tf.div(boxHW, 2)), modelSize);

		const boxes = tf.concat([...tf.split(boxMins, 2, 4), ...tf.split(boxMaxes, 2, 4)], 4).reshape([batch, numBoxes, 4]);

		boxConfidence = tf.sigmoid(boxConfidence);
		boxClassProbs = tf.softmax(boxClassProbs);

		const classProbs = tf.mul(boxConfidence, boxClassProbs).reshape([batch, numBoxes, numClasses]);

		return [boxes, classProbs];
	}

	async postProcessBoxes(rawBoxes, rawScores) {
		const scores = tf.max(rawScores, -1);
		const classes = tf.argMax(rawScores, -1);

		const indiceTensor = await tf.image.nonMaxSuppressionAsync(rawBoxes, scores, this.maxOutput, this.iouThreshold, this.classProbThreshold);
		const filteredBoxes = tf.gather(rawBoxes, indiceTensor).array(); // [batch, n, 4]
		const filteredScores = tf.gather(scores, indiceTensor).array(); // [batch, n,  ]
		const filteredClasses = tf.gather(classes, indiceTensor).array(); // [batch, n,  ]
		return Promise.all([filteredBoxes, filteredScores, filteredClasses]);
	}

	/**
	 * The final phase in the post processing that outputs the final `Detection[]`
	 * @param finalBoxes an array containing the raw box information
	 * @param imageDims a `Shape` containing the original image dimensions `[height, width]`
	 * @param inputDims a `Shape` containing the model input dimensions `[height, width]`
	 * @return a `DetectorOutput` with the final collected boxes
	 */
	 createDetectionArray(boxes, scores, classes, modelSize, originalInputSize) {
		const numDetections = classes.length; // || scores.length;
		const detections = [];
		for (let i = 0; i < numDetections; i += 1) {
			// debugger;
			const topY = boxes[i][0];
			const topX = boxes[i][1];
			const bottomY = boxes[i][2];
			const bottomX = boxes[i][3];

			const w = bottomX - topX;
			const h = bottomY - topY;

			const scaleX = originalInputSize[1] / modelSize[1]; // width
			const scaleY = originalInputSize[0] / modelSize[0]; // height

			const classIndex = classes[i];
			const label = CLASS_NAMES[classIndex];

			const score = scores[i];

			const detection = {
				raw: {
					x : topX,
					y : topY,
					w,
					h,
					imageWidth : modelSize[1],
					imageHeight : modelSize[0],
				},

				labelIndex: classIndex,
				label,
				confidence : score,

				x: topX * scaleX,
				y: topY * scaleY,
				w: w * scaleX,
				h: h * scaleY,
				
				imageWidth : originalInputSize[1],
				imageHeight : originalInputSize[0],
			};
			detections.push(detection);
		}		
		return detections;
	}
}

const YOLO = (videoOr, optionsOr, cb) => {
	let video = null;
	let options = {};
	let callback = cb;

	if (videoOr instanceof HTMLVideoElement) {
		video = videoOr;
	} else if (typeof videoOr === 'object' && videoOr.elt instanceof HTMLVideoElement) {
		video = videoOr.elt; // Handle p5.js image
	} else if (typeof videoOr === 'function') {
		callback = videoOr;
	} else if (typeof videoOr === 'object') {
		options = videoOr;
	}

	if (typeof optionsOr === 'object') {
		options = optionsOr;
	} else if (typeof optionsOr === 'function') {
		callback = optionsOr;
	}

	return new YOLOBase(video, options, callback);
};

export default YOLO;
