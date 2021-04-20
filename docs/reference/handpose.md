# Handpose


<center>
    <img style="display:block; max-height:20rem" alt="A GIF of a person waving their hand in front of a camera. Green dots are drawn over different locations on their palm and fingers–demonstrating the capabilities of the Handpose model." src="_media/reference__header-handpose.gif">
</center>


## Description

Handpose is a machine-learning model that allows for palm detection and hand-skeleton finger tracking in the browser. It can detect a maximum of one hand at a time and provides 21 3D hand keypoints that describe important locations on the palm and fingers.

The ml5.js Handpose model is ported from the [TensorFlow.js Handpose implementation](https://github.com/tensorflow/tfjs-models/tree/master/handpose).

## Quickstart

```js
let predictions = [];
const video = document.getElementById('video');

// Create a new handpose method
const handpose = ml5.handpose(video, modelLoaded);

// When the model is loaded
function modelLoaded() {
  console.log('Model Loaded!');
}

// Listen to new 'hand' events
handpose.on('hand', results => {
  predictions = results;
});
```


## Usage

### Initialize
You can initialize ml5.handpose with an optional `video`, configuration `options` object, or a `callback` function.
```js
const handpose = ml5.handpose(?video, ?options, ?callback);
```

#### Parameters
* **video**: OPTIONAL. Optional HTMLVideoElement input to run predictions on.
* **options**: OPTIONAL. A object that contains properties that effect the Handpose model accuracy, results, etc. See documentation on the available options in [TensorFlow's Handpose documentation](https://github.com/tensorflow/tfjs-models/tree/master/handpose#parameters-for-handposeload).
```js
const options = {
  flipHorizontal: false, // boolean value for if the video should be flipped, defaults to false
  maxContinuousChecks: Infinity, // How many frames to go without running the bounding box detector. Defaults to infinity, but try a lower value if the detector is consistently producing bad predictions.
  detectionConfidence: 0.8, // Threshold for discarding a prediction. Defaults to 0.8.
  scoreThreshold: 0.75, // A threshold for removing multiple (likely duplicate) detections based on a "non-maximum suppression" algorithm. Defaults to 0.75
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
> *Object*. The Handpose model.
***

***
#### .modelReady
> *Boolean*. Truthy value indicating the model has loaded.
***

### Methods

***
#### .predict()
> A function that returns the results of a single hand detection prediction.

  ```js
  handpose.predict(inputMedia, callback);
  ```

📥 **Inputs**
* **inputMedia**: REQUIRED. An HMTL or p5.js image, video, or canvas element that you'd like to run a single prediction on.

* **callback**: OPTIONAL.  A callback function to handle new hand detection predictions. For example:

  ```js
  handpose.predict(inputMedia, results => {
    // do something with the results
    console.log(results);
  });
  ```

📤 **Outputs**

* **Array**: Returns an array of objects describing each detected hand. You can see all of the supported annotation in [the Tensorflow source code](https://github.com/tensorflow/tfjs-models/blob/master/handpose/src/keypoints.ts).

  ```js
  [
      {
        handInViewConfidence: 1, // The probability of a hand being present.
        boundingBox: { // The bounding box surrounding the hand.
          topLeft: [162.91, -17.42],
          bottomRight: [548.56, 368.23],
        },
        landmarks: [ // The 3D coordinates of each hand landmark.
          [472.52, 298.59, 0.00],
          [412.80, 315.64, -6.18],
          ...
        ],
        annotations: { // Semantic groupings of the `landmarks` coordinates.
          thumb: [
            [412.80, 315.64, -6.18]
            [350.02, 298.38, -7.14],
            ...
          ],
          ...
        }
      }
    ]
  ```

***

#### .on('hand', ...)
> An event listener that returns the results when a new hand detection prediction occurs.

  ```js
  handpose.on('hand', callback);
  ```

📥 **Inputs**

* **callback**: REQUIRED.  A callback function to handle new hand detection predictions. For example:

  ```js
  handpose.on('hand', results => {
    // do something with the results
    console.log(results);
  });
  ```

📤 **Outputs**

* **Array**: Returns an array of objects describing each detected hand as an array of objects exactly like the output of the `.predict()` method described above.


## Examples

**p5.js**
* [Handpose_Image](https://github.com/ml5js/ml5-library/tree/main/examples/p5js/Handpose/Handpose_Image)
* [Handpose_Webcam](https://github.com/ml5js/ml5-library/tree/main/examples/p5js/Handpose/Handpose_Webcam)

**p5 web editor**
* [Handpose_Image](https://editor.p5js.org/ml5/sketches/Handpose_Image)
* [Handpose_Webcam](https://editor.p5js.org/ml5/sketches/Handpose_Webcam)

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

* [/src/Handpose](https://github.com/ml5js/ml5-library/tree/main/src/Handpose)
