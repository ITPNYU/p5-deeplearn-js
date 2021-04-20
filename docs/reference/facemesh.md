# Facemesh


<center>
    <img style="display:block; max-height:20rem" alt="A screenshot of a video feed where a person sits at their chair inside of a bedroom while green dots are drawn over different locations on their face." src="_media/reference__header-facemesh.jpg">
</center>


## Description

Facemesh is a machine-learning model that allows for facial landmark detection in the browser. It can detect multiple faces at once and provides 486 3D facial landmarks that describe the geometry of each face. Facemesh works best when the faces in view take up a large percentage of the image or video frame and it may struggle with small/distant faces.

The ml5.js Facemesh model is ported from the [TensorFlow.js Facemesh implementation](https://github.com/tensorflow/tfjs-models/tree/master/facemesh).

## Quickstart

```js
let predictions = [];
const video = document.getElementById('video');

// Create a new facemesh method
const facemesh = ml5.facemesh(video, modelLoaded);

// When the model is loaded
function modelLoaded() {
  console.log('Model Loaded!');
}

// Listen to new 'face' events
facemesh.on('face', results => {
  predictions = results;
});
```


## Usage

### Initialize
You can initialize ml5.facemesh with an optional `video`, configuration `options` object, or a `callback` function.
```js
const facemesh = ml5.facemesh(?video, ?options, ?callback);
```

#### Parameters
* **video**: OPTIONAL. Optional HTMLVideoElement input to run predictions on.
* **options**: OPTIONAL. A object that contains properties that effect the Facemesh model accuracy, results, etc. See documentation on the available options in [TensorFlow's Facemesh documentation](https://github.com/tensorflow/tfjs-models/tree/master/facemesh#parameters-for-facemeshload).
  ```js
  const options = {
  flipHorizontal: false, // boolean value for if the video should be flipped, defaults to false
  maxContinuousChecks: 5, // How many frames to go without running the bounding box detector. Only relevant if maxFaces > 1. Defaults to 5.
  detectionConfidence: 0.9, // Threshold for discarding a prediction. Defaults to 0.9.
  maxFaces: 10, // The maximum number of faces detected in the input. Should be set to the minimum number for performance. Defaults to 10.
  scoreThreshold: 0.75, // A threshold for removing multiple (likely duplicate) detections based on a "non-maximum suppression" algorithm. Defaults to 0.75.
  iouThreshold: 0.3, // A float representing the threshold for deciding whether boxes overlap too much in non-maximum suppression. Must be between [0, 1]. Defaults to 0.3.
  }
  ```

* **callback**: OPTIONAL. A function that is called once the model has loaded.

### Properties
***
#### .video
> *Object*. HTMLVideoElement if given in the constructor. Otherwise it is null.
***

***
#### .config
> *Object*. containing all of the configuration options passed into the model. 
***

***
#### .model
> *Object*. The Facemesh model.
***

***
#### .modelReady
> *Boolean*. Truthy value indicating the model has loaded.
***

### Methods

***
#### .predict()
> A function that returns the results of a single face detection prediction.

  ```js
  facemesh.predict(inputMedia, callback);
  ```

📥 **Inputs**
* **inputMedia**: REQUIRED. An HMTL or p5.js image, video, or canvas element that you'd like to run a single prediction on.

* **callback**: OPTIONAL.  A callback function to handle new face detection predictions. For example:

  ```js
  facemesh.predict(inputMedia, results => {
    // do something with the results
    console.log(results);
  });
  ```

📤 **Outputs**

* **Array**: Returns an array of objects describing each detected face. See the [Facemesh keypoints map](https://github.com/tensorflow/tfjs-models/tree/master/facemesh#keypoints) to determine how the keypoint related to facial landmarks.

  ```js
  [
      {
          faceInViewConfidence: 1, // The probability of a face being present.
          boundingBox: { // The bounding box surrounding the face.
              topLeft: [232.28, 145.26],
              bottomRight: [449.75, 308.36],
          },
          mesh: [ // The 3D coordinates of each facial landmark.
              [92.07, 119.49, -17.54],
              [91.97, 102.52, -30.54],
              ...
          ],
          scaledMesh: [ // The 3D coordinates of each facial landmark, normalized.
              [322.32, 297.58, -17.54],
              [322.18, 263.95, -30.54]
          ],
          annotations: { // Semantic groupings of the `scaledMesh` coordinates.
              silhouette: [
              [326.19, 124.72, -3.82],
              [351.06, 126.30, -3.00],
              ...
              ],
              ...
          }
      }
  ]
  ```

***

#### .on('face', ...)
> An event listener that returns the results when a new face detection prediction occurs.

  ```js
  facemesh.on('face', callback);
  ```

📥 **Inputs**

* **callback**: REQUIRED.  A callback function to handle new face detection predictions. For example:

  ```js
  facemesh.on('face', results => {
    // do something with the results
    console.log(results);
  });
  ```

📤 **Outputs**

* **Array**: Returns an array of objects describing each detected face as an array of objects exactly like the output of the `.predict()` method described above. See the [Facemesh keypoints map](https://github.com/tensorflow/tfjs-models/tree/master/facemesh#keypoints) to determine how the keypoint related to facial landmarks.


## Examples

**p5.js**
* [Facemesh_Image](https://github.com/ml5js/ml5-library/tree/main/examples/p5js/Facemesh/Facemesh_Image)
* [Facemesh_Webcam](https://github.com/ml5js/ml5-library/tree/main/examples/p5js/Facemesh/Facemesh_Webcam)

**p5 web editor**
* [Facemesh_Image](https://editor.p5js.org/ml5/sketches/Facemesh_Image)
* [Facemesh_Webcam](https://editor.p5js.org/ml5/sketches/Facemesh_Webcam)

## Demo

No demos yet - contribute one today!

## Tutorials

No tutorials yet - contribute one today!

## Model and Data Provenance
> A project started by [Ellen Nickles](https://github.com/ellennickles/)

Coming soon!

## Acknowledgements

**Contributors**:
  * Ported to ml5.js by [Bomani Oseni McClendon](https://bomani.rip/).

## Source Code

* [/src/Facemesh](https://github.com/ml5js/ml5-library/tree/main/src/Facemesh)
