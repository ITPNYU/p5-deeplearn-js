let handpose;
let predictions = [];
let img;

// load the image before the main program starts
function preload(){
  img = loadImage("data/hand.jpg");
}

function setup() {
  // Create a canvas that's at least the size of the image.
  createCanvas(400, 350);
  // call modelReady() when it is loaded
  handpose = ml5.handpose(modelReady);

  frameRate(1); // set the frameRate to 1 since we don't need it to be running quickly in this case
}

// when poseNet is ready, do the detection
function modelReady() {
  console.log("Model ready!");
  
  // when the predict function is called, tell 
  // handpose what to do with the results.
  // in this case we assign the results to our global
  // predictions variable
  handpose.on("hand", results => {
    predictions = results;
  });

  handpose.predict(img);
}

// draw() will not show anything until poses are found
function draw() {
  if (predictions.length > 0) {
    image(img, 0, 0, width, height);
    drawKeypoints();
    noLoop(); // stop looping when the poses are estimated
  }
}

// A function to draw ellipses over the detected keypoints
function drawKeypoints() {
  for (let i = 0; i < predictions.length; i += 1) {
    const prediction = predictions[i];
    for (let j = 0; j < prediction.landmarks.length; j += 1) {
      const keypoint = prediction.landmarks[j];
      fill(0, 255, 0);
      noStroke();
      ellipse(keypoint[0], keypoint[1], 10, 10);
    }
  }
}