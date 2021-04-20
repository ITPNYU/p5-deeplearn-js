let facemesh;
let predictions = [];
let img;

function setup() {
  // Create a canvas that's at least the size of the image.
  createCanvas(270, 340);

  // create an image using the p5 dom library
  // call modelReady() when it is loaded
  img = createImg("data/face.png", imageReady);
  // set the image size to the size of the canvas

  img.hide(); // hide the image in the browser
  frameRate(1); // set the frameRate to 1 since we don't need it to be running quickly in this case
}

// when the image is ready, then load up poseNet
function imageReady() {
  facemesh = ml5.facemesh(modelReady);

  facemesh.on("face", results => {
    predictions = results;
  });
}

// when poseNet is ready, do the detection
function modelReady() {
  console.log("Model ready!");
  facemesh.predict(img);
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
    const keypoints = predictions[i].scaledMesh;

    // Draw facial keypoints.
    for (let j = 0; j < keypoints.length; j += 1) {
      const [x, y] = keypoints[j];

      fill(0, 255, 0);
      ellipse(x, y, 5, 5);
    }
  }
}
