// Copyright (c) 2018 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

import * as tf from "@tensorflow/tfjs";
import { getTopKClassesFromTensor } from "../utils/gettopkclasses";
import IMAGENET_CLASSES_DARKNET from "../utils/IMAGENET_CLASSES_DARKNET";

const DEFAULTS = {
  DARKNET_URL:
    "https://cdn.jsdelivr.net/gh/ml5js/ml5-data-and-models@master/models/darknetclassifier/darknetreference/model.json",
  DARKNET_TINY_URL:
    "https://cdn.jsdelivr.net/gh/ml5js/ml5-data-and-models@master/models/darknetclassifier/darknettiny/model.json",
  IMAGE_SIZE_DARKNET: 256,
  IMAGE_SIZE_DARKNET_TINY: 224,
};

function preProcess(img, size) {
  let image;
  if (!(img instanceof tf.Tensor)) {
    if (
      img instanceof HTMLImageElement ||
      img instanceof HTMLVideoElement ||
      img instanceof HTMLCanvasElement ||
      img instanceof ImageData
    ) {
      image = tf.browser.fromPixels(img);
    } else if (
      typeof img === "object" &&
      (img.elt instanceof HTMLImageElement ||
        img.elt instanceof HTMLVideoElement ||
        img.elt instanceof HTMLCanvasElement ||
        img.elt instanceof ImageData)
    ) {
      image = tf.browser.fromPixels(img.elt); // Handle p5.js image and video.
    }
  } else {
    image = img;
  }
  const normalized = image.toFloat().div(tf.scalar(255));
  let resized = normalized;
  if (normalized.shape[0] !== size || normalized.shape[1] !== size) {
    const alignCorners = true;
    resized = tf.image.resizeBilinear(normalized, [size, size], alignCorners);
  }
  const batched = resized.reshape([1, size, size, 3]);
  return batched;
}

export class Darknet {
  constructor(version) {
    this.version = version;
    switch (this.version) {
      case "reference":
        this.imgSize = DEFAULTS.IMAGE_SIZE_DARKNET;
        break;
      case "tiny":
        this.imgSize = DEFAULTS.IMAGE_SIZE_DARKNET_TINY;
        break;
      default:
        break;
    }
  }

  async load() {
    switch (this.version) {
      case "reference":
        this.model = await tf.loadLayersModel(DEFAULTS.DARKNET_URL);
        break;
      case "tiny":
        this.model = await tf.loadLayersModel(DEFAULTS.DARKNET_TINY_URL);
        break;
      default:
        break;
    }

    // Warmup the model.
    const result = tf.tidy(() => this.model.predict(tf.zeros([1, this.imgSize, this.imgSize, 3])));
    await result.data();
    result.dispose();
  }

  async classify(img, topk = 3) {
    const logits = tf.tidy(() => {
      const imgData = preProcess(img, this.imgSize);
      const predictions = this.model.predict(imgData);
      return tf.softmax(predictions);
    });
    const classes = await getTopKClassesFromTensor(logits, topk, IMAGENET_CLASSES_DARKNET);
    logits.dispose();
    return classes;
  }
}

export async function load(modelConfig) {
  const { version } = modelConfig
  if (version !== "reference" && version !== "tiny") {
    throw new Error("Please select a version: darknet-reference or darknet-tiny");
  }

  const darknet = new Darknet(version);
  await darknet.load();
  return darknet;
}
