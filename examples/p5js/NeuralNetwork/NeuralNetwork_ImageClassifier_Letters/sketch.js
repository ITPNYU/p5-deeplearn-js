// Copyright (c) 2019 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/* ===
ml5 Example
Image classification using Convolutional Neural Network
This example uses a callback pattern to create the classifier
=== */

let nn;
const IMAGE_WIDTH = 64;
const IMAGE_HEIGHT = 64;
const IMAGE_CHANNELS = 4;

const images = [];
let testA;

function preload() {
  for (let i = 1; i < 7; i += 1) {
    const a = loadImage(`images/A_0${i}.png`);
    const b = loadImage(`images/B_0${i}.png`);
    images.push({ image: a, label: 'A' });
    images.push({ image: b, label: 'B' });
  }
  testA = loadImage(`images/A_test.png`);
}

function setup() {
  createCanvas(128, 128);
  image(testA, 0, 0, width, height);

  const options = {
    inputs: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
    task: 'imageClassification',
    debug: 'true',
  };

  // construct the neural network
  nn = ml5.neuralNetwork(options);

  // add data
  for (let i = 0; i < images.length; i += 1) {
    nn.addData({ image: images[i].image }, { label: images[i].label });
  }

  // normalize data
  nn.normalizeData();

  nn.train({ epochs: 20 }, finishedTraining);
}

function finishedTraining() {
  console.log('finished training');
  // method 1: you can pass in an object with a matching key and the p5 image
  nn.classify({ image: testA }, gotResults);
}

function gotResults(err, results) {
  if (err) {
    console.log(err);
    return;
  }
  console.log(results);
  const percent = 100 * results[0].confidence;
  createP(`${results[0].label} ${nf(percent, 2, 1)}%`);
}
