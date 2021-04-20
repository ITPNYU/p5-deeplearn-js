let img;
let poseNet;

function preload() {
  // load an image for pose detection
  img = loadImage('data/runner.jpg');
}

function setup() {
  createCanvas(640, 360);
  image(img, 0, 0);
  poseNet = ml5.poseNet(modelReady);
}

// when poseNet is ready, do the detection
function modelReady() {
  select('#status').html('Model Loaded');
  // If/When a pose is detected, poseNet.on('pose', ...) will be listening for the detection results 
  poseNet.on('pose', function (poses) {
    if (poses.length > 0) {
      drawSkeleton(poses);
      drawKeypoints(poses);
    }
  });
  // When the model is ready, run the singlePose() function...
  poseNet.singlePose(img);
}

// The following comes from https://ml5js.org/docs/posenet-webcam
// A function to draw ellipses over the detected keypoints
function drawKeypoints(poses) {
  // Loop through all the poses detected
  for (let i = 0; i < poses.length; i++) {
    // For each pose detected, loop through all the keypoints
    let pose = poses[i].pose;
    for (let j = 0; j < pose.keypoints.length; j++) {
      // A keypoint is an object describing a body part (like rightArm or leftShoulder)
      let keypoint = pose.keypoints[j];
      // Only draw an ellipse is the pose probability is bigger than 0.2
      if (keypoint.score > 0.2) {
        fill(255);
        stroke(20);
        strokeWeight(4);
        ellipse(round(keypoint.position.x), round(keypoint.position.y), 8, 8);
      }
    }
  }
}

// A function to draw the skeletons
function drawSkeleton(poses) {
  // Loop through all the skeletons detected
  for (let i = 0; i < poses.length; i++) {
    let skeleton = poses[i].skeleton;
    // For every skeleton, loop through all body connections
    for (let j = 0; j < skeleton.length; j++) {
      let partA = skeleton[j][0];
      let partB = skeleton[j][1];
      stroke(255);
      strokeWeight(1);
      line(partA.position.x, partA.position.y, partB.position.x, partB.position.y);
    }
  }
}