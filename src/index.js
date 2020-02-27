// Copyright (c) 2018 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import pitchDetection from './PitchDetection/';
import imageClassifier from './ImageClassifier/';
import soundClassifier from './SoundClassifier/';
import KNNClassifier from './KNNClassifier/';
import featureExtractor from './FeatureExtractor/';
import word2vec from './Word2vec/';
import YOLO from './ObjectDetector/YOLO';
import objectDetector from './ObjectDetector';
import poseNet from './PoseNet';
import * as imageUtils from './utils/imageUtilities';
import * as drawUtils from './utils/drawDetections';

import styleTransfer from './StyleTransfer/';
import charRNN from './CharRNN/';
import pix2pix from './Pix2pix/';
import sketchRNN from './SketchRNN';
import uNet from './UNET';
import CVAE from './CVAE';
import DCGAN from './DCGAN';
import preloadRegister from './utils/p5PreloadHelper';
import { version } from '../package.json';
import sentiment from './Sentiment';
import bodyPix from './BodyPix';
import neuralNetwork from './NeuralNetwork';
import faceApi from './FaceApi';
import kmeans from './KMeans';
import p5Utils from './utils/p5Utils';
import communityStatement from './utils/community';

const withPreload = {
  charRNN,
  CVAE,
  DCGAN,
  featureExtractor,
  imageClassifier,
  kmeans,
  soundClassifier,
  pitchDetection,
  pix2pix,
  poseNet,
  sketchRNN,
  styleTransfer,
  word2vec,
  YOLO,
  objectDetector,
  uNet,
  sentiment,
  bodyPix,
  faceApi,
};

// call community statement on load
(() => {
  communityStatement();
})();

module.exports = Object.assign({p5Utils}, preloadRegister(withPreload), {
  KNNClassifier,
  communityStatement,
  ...imageUtils,
  ...drawUtils,
  tf,
  tfvis,
  version,
  neuralNetwork,
});
