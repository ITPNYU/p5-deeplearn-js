let handpose;
let video;
let hands = [];

function setup() {
  createCanvas(640, 480);
  video = createCapture(VIDEO);
  video.size(width, height);

  handpose = ml5.handpose(video, modelReady);

  // This sets up an event that fills the global variable "predictions"
  // with an array every time new hand poses are detected
  handpose.on("predict", function(results) {
    hands = results;
  });

  // Hide the video element, and just show the canvas
  video.hide();
}

function modelReady() {
  console.log("Model ready!");
}

function draw() {
  image(video, 0, 0, width, height);

  // If there is a hand
  if (hands.length > 0) {
    // Individual parts are in "annotations"
    let thumb = hands[0].annotations.thumb;
    fill(0, 255, 0);
    noStroke();
    // The top of the thumb is index 3
    ellipse(thumb[3][0], thumb[3][1], 24);
    let index = hands[0].annotations.indexFinger;
    fill(0, 0, 255);
    noStroke();
    // The top of the index finger is index 3
    ellipse(index[3][0], index[3][1], 24);
  }
}
