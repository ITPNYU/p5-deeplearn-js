let bodypix;
let video;
let segmentation;

const options = {
  outputStride: 16, // 8, 16, or 32, default is 16
  segmentationThreshold: 0.5, // 0 - 1, defaults to 0.5
};

function preload() {
    bodypix = ml5.bodyPix(options);
}

function setup() {
  createCanvas(320, 240);
  // load up your video
  video = createCapture(VIDEO, () => {
    // video.size(width, height);
    video.hide(); // Hide the video element, and just show the canvas
    segment();
  });
}

function segment() {
  bodypix.segment(video, gotResults);
}

function gotResults(err, result) {
  if (err) {
    console.log(err);
    return;
  }

  segmentation = result;

  background(0);
  image(segmentation.backgroundMask, 0, 0, width, height);

  segment();
}
