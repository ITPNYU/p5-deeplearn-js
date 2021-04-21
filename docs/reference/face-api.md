# Face-Api

<center>
    <img style="display:block; max-height:20rem" alt="face landmark detection" src="_media/reference__header-faceapi.png">
</center>


## Description

ml5.js has created an API to [face-api.js](https://github.com/justadudewhohacks/face-api.js/blob/master/README.md) that allows you to access face and face landmark detection.  

The ml5.js implementation of face-api does not support `expressions`, `age` or `gender` estimation.

## Quickstart

```js
const detectionOptions = {
  withLandmarks: true,
  withDescriptors: false,
};
// Initialize the magicFeature
const faceapi = ml5.faceApi(detectionOptions, modelLoaded);

// When the model is loaded
function modelLoaded() {
  console.log('Model Loaded!');

  // Make some sparkles
  faceapi.detect(myImage, (err, results) => {
    console.log(results);
  });
}


```


## Usage

### Initialize

```js
const faceapi = ml5.faceApi(videoOrOptionsOrCallback, optionsOrCallback?, callback?);
```

#### Parameters
* **videoOrOptionsOrCallback**: REQUIRED. Notice there is no question mark in front of the input.
* **optionsOrCallback**: OPTIONAL. Notice the `?` indicates an optional parameter.
* **callback**: OPTIONAL. A description of some kind of object with some properties. Notice the `?` indicates an optional parameter.

  ```js
  {
    withLandmarks: true,
    withDescriptors: true,
    minConfidence: 0.5,
    MODEL_URLS: {
      Mobilenetv1Model: 'https://raw.githubusercontent.com/ml5js/ml5-data-and-models/main/models/faceapi/ssd_mobilenetv1_model-weights_manifest.json',
      FaceLandmarkModel: 'https://raw.githubusercontent.com/ml5js/ml5-data-and-models/main/models/faceapi/face_landmark_68_model-weights_manifest.json',
      FaceLandmark68TinyNet: 'https://raw.githubusercontent.com/ml5js/ml5-data-and-models/main/models/faceapi/face_landmark_68_tiny_model-weights_manifest.json',
      FaceRecognitionModel: 'https://raw.githubusercontent.com/ml5js/ml5-data-and-models/main/models/faceapi/face_recognition_model-weights_manifest.json',
    },
  };
  ```

### Properties

***
#### .video
> **Object**: An HTML5Video object if one is passed in the constructor
***

***
#### .model
> **Object**: the model
***

***
#### .modelReady
> **Boolean**: true when model is ready and loaded, false when it is not
***

***
#### .config
> **Object** object of configurations of the model
***

***
#### .ready
> **Boolean**: true when model is ready and loaded, false when it is not
***



### Methods


***
#### .detect()
> Given an image, will run face detection. If a video was passed in the constructor, then it only a callback is necessary to handle the results.

```js
faceapi.detect(optionsOrCallback, configOrCallback, callback);
```

📥 **Inputs**

* **optionsOrCallback**: Required. Image | Object | Function. If given an image, this is the image that the face detection will be applied. If given an object, it is to set the configuration. If given a callback, this is to handle the results of the `.detect()` method.
* **configOrCallback**: Optional. Object | Function. If an image is given as the first parameter, then this will either be an object or a function. If given an object, it is to set the configuration. If given a callback, this is to handle the results of the `.detect()` method.
* **callback**: Optional. Function. A function to handle the results of `.makeSparkles()`. Likely a function to do something with the results of makeSparkles.

📤 **Outputs**

* **Array**: Returns an array of objects. Each object contains `{alignedRect, detection, landmarks, unshiftedLandmarks}`.

***

***
#### .detectSingle()
> Given an image, will run face detection. If a video was passed in the constructor, then it only a callback is necessary to handle the results. `.detectSingle()` is more accurate.

```js
faceapi.detectSingle(optionsOrCallback, configOrCallback, callback);
```

📥 **Inputs**
* **optionsOrCallback**: Required. Image | Object | Function. If given an image, this is the image that the face detection will be applied. If given an object, it is to set the configuration. If given a callback, this is to handle the results of the `.detect()` method.
* **configOrCallback**: Optional. Object | Function. If an image is given as the first parameter, then this will either be an object or a function. If given an object, it is to set the configuration. If given a callback, this is to handle the results of the `.detect()` method.
* **callback**: Optional. Function. A function to handle the results of `.makeSparkles()`. Likely a function to do something with the results of makeSparkles.

📤 **Outputs**

* **Array**: Returns an array of objects. Each object contains `{alignedRect, detection, landmarks, unshiftedLandmarks}`.

***




## Examples

**p5.js**
* [FaceApi_Image_Landmarks](https://github.com/ml5js/ml5-library/tree/main/examples/p5js/FaceApi/FaceApi_Image_Landmarks)
* [FaceApi_Video_Landmarks](https://github.com/ml5js/ml5-library/tree/main/examples/p5js/FaceApi/FaceApi_Video_Landmarks)
* [FaceApi_Video_Landmarks_LocalModels](https://github.com/ml5js/ml5-library/tree/main/examples/p5js/FaceApi/FaceApi_Video_Landmarks_LocalModels)

**p5 web editor**
* [FaceApi_Image_Landmarks](https://editor.p5js.org/ml5/sketches/FaceApi_Image_Landmarks)
* [FaceApi_Video_Landmarks](https://editor.p5js.org/ml5/sketches/FaceApi_Video_Landmarks)

**plain javascript**
* [FaceApi_Image_Landmarks](https://github.com/ml5js/ml5-library/tree/main/examples/javascript/FaceApi/FaceApi_Image_Landmarks/)
* [FaceApi_Video_Landmarks](https://github.com/ml5js/ml5-library/tree/main/examples/javascript/FaceApi/FaceApi_Video_Landmarks/)
* [FaceApi_Video_Landmarks_LocalModels](https://github.com/ml5js/ml5-library/tree/main/examples/javascript/FaceApi/FaceApi_Video_Landmarks_LocalModels/)

## Demo

* [FaceApi_Video_Landmarks Demo](https://git.xho.to/faceapi-demo/)

## Tutorials

No tutorials yet - contribute one today!


## Model and Data Provenance
> A project started by [Ellen Nickles](https://github.com/ellennickles/)

### Models Overview

FaceApi provides the following five pre-trained model options for different tasks: MobilenetV1, TinyFaceDetector, FaceLandmarkModel, FaceLandmark68TinyNet, and FaceRecognitionModel.

Each model is a type of convolutional neural network (CNN). A CNN finds patterns in the pixels of images (e.g. horizontal or vertical edges), and through successive layers of computation finds sets of patterns to identify more complex patterns (e.g. corners or circles), eventually detecting intricate patterns that it predicts belong to a particular category (e.g. a face or a point on a face). The categories depend on how images in the model’s training dataset are labeled. In the case of the face recognition model, the “model is not limited to the set of faces used for training, meaning you can use it for face recognition of any person” because the model has been trained to  “determine the similarity of two arbitrary faces by comparing their face descriptors (source).” 


#### MobilenetV1 and TinyFaceDetector - Model Biography

- **Description**
  - These models detect faces in images and video. TinyFaceDetector is smaller, faster, and mobile-friendly, but potentially less accurate.
- **Developer and Year**
  - These models are actively developed for the browser by Vincent Mühler, beginning in 2018. Technical information about models’ architectures are available in the face-api.js GitHub repository.
  - Mühler writes that the model’s weights were retrieved from a GitHub repository hosted by yeephycho, who also used the WIDER FACE dataset. (Weights are the parameters that machine learning neural networks use to perform a task, such as to predict features, or patterns, in a dataset. Weights are the values that exist in the connections between the layers of a network. At each layer, a mathematical function is applied to these values to output new values for the subsequent layer and eventually generate a final prediction for the result.)
- **Purpose and Intended Users**
  - TBD
- **Hosted Location**
  - These models are hosted by ml5.
- **ml5 Contributor and Year**
  - Ported by Joey Lee in 2019
- **References**
  - Developer [Vincent Mühler](https://github.com/justadudewhohacks)
  - ml5 Contributor [Joey Lee](https://github.com/joeyklee)
  - GitHub Repository [face-api.js](https://github.com/justadudewhohacks/face-api.js)
  - GitHub Repository [yeephycho’s Tensorflow Face Detector](https://github.com/yeephycho/tensorflow-face-detection)
  - Article [face-api.js — JavaScript API for Face Recognition in the Browser with tensorflow.js](https://itnext.io/face-api-js-javascript-api-for-face-recognition-in-the-browser-with-tensorflow-js-bcc2a6c4cf07)

#### MobilenetV1 and TinyFaceDetector - Data Biography

- **Description**
  - MobileNetV1 was trained on images from the WIDER FACE dataset. From the WIDER FACE website, it contains “32,203 images and label[d] 393,703 faces with a high degree of variability in scale, pose and occlusion as depicted in the sample images.”
  - Mühler writes that TinyFaceDetector was trained on a custom dataset of ~14K images labeled with bounding boxes.
- **Source**
  - For MobileNetV1, according to the WIDER FACE website, the WIDER FACE dataset uses publicly available images from WIDE dataset. 
- **Collector and Year**
  - The WIDER FACE dataset is hosted by Shuo Yang.
- **Collection Method**
  - TBD
- **Purpose and Intended Users**
  - TBD
- **References**
  - Developer [Vincent Mühler](https://github.com/justadudewhohacks)
  - GitHub Repository [face-api.js](https://github.com/justadudewhohacks/face-api.js)
  - Website [WIDER FACE Dataset](http://shuoyang1213.me/WIDERFACE/)
  - Website [Shuo Yang](http://shuoyang1213.me/)

#### FaceLandmarkModel and FaceLandmark68TinyNet - Model Biography

- **Description**
  - These models detect 68 facial landmarks, or points, on a detected face. FaceLandmark68TinyNet is the smaller of the two. 
- **Developer and Year**
  - These models are actively developed for the browser by Vincent Mühler, beginning in 2018. Technical information about models’ architectures are available in the face-api.js GitHub repository.
- **Purpose and Intended Users**
  - TBD
- **Hosted Location**
  - Same as above
- **ml5 Contributor and Year**
  - Same as above
- **References**
  - Developer [Vincent Mühler](https://github.com/justadudewhohacks)
  - ml5 Contributor [Joey Lee](https://github.com/joeyklee)
  - GitHub Repository [face-api.js](https://github.com/justadudewhohacks/face-api.js)
  - Article [face-api.js — JavaScript API for Face Recognition in the Browser with tensorflow.js](https://itnext.io/face-api-js-javascript-api-for-face-recognition-in-the-browser-with-tensorflow-js-bcc2a6c4cf07)

#### FaceLandmarkModel and FaceLandmark68TinyNet - Data Biography

- **Description**
  - These models were trained on a dataset of ~35K face images labeled with 68 facial points.
- **Source**
  - TBD
- **Collector and Year**
  - TBD
- **Collection Method**
  - TBD
- **Purpose and Intended Users**
  - TBD
- **References**
  - GitHub Repository [face-api.js](https://github.com/justadudewhohacks/face-api.js)

#### FaceRecognitionModel - Model Biography

- **Description**
  - This model computes a person’s facial characteristics in order to detect or “recognize” them in another image or video.
- **Developer and Year**
  - These models are actively developed for the browser by Vincent Mühler, beginning in 2018. Technical information about models’ architectures are available in the face-api.js GitHub repository.
  - Mühler writes that weights for the FaceRecognition Model were retrieved from Davis King. (Weights are the parameters that machine learning neural networks use to perform a task, such as to predict features, or patterns, in a dataset. Weights are the values that exist in the connections between the layers of a network. At each layer, a mathematical function is applied to these values to output new values for the subsequent layer and eventually generate a final prediction for the result.)
- **Purpose and Intended Users**
  - TBD
- **Hosted Location**
  - Same as above
- **ml5 Contributor and Year**
  - Same as above
- **References**
  - Developers [Vincent Mühler](https://github.com/justadudewhohacks) and [Davis King](https://github.com/davisking)
  - ml5 Contributor [Joey Lee](https://github.com/joeyklee)
  - GitHub Repository [face-api.js](https://github.com/justadudewhohacks/face-api.js)
  - GitHub Repository [David King’s dlib-models](https://github.com/davisking/dlib-models)
  - Article [face-api.js — JavaScript API for Face Recognition in the Browser with tensorflow.js](https://itnext.io/face-api-js-javascript-api-for-face-recognition-in-the-browser-with-tensorflow-js-bcc2a6c4cf07)

#### FaceRecognitionModel - Data Biography

- **Description**
  - According to King’s GitHub repository, the dataset consists of three million images.
- **Source**
  - The images were compiled those scraped from the internet, the FaceScrub Dataset, and the VGG Face Dataset.
- **Collector and Year**
  - TBD
- **Collection Method**
  - TBD
- **Purpose and Intended Users**
  - TBD
- **References**
  - Website [FaceScrub Dataset](http://vintage.winklerbros.net/facescrub.html)
  - Website [VGG Face Dataset](http://www.robots.ox.ac.uk/~vgg/data/vgg_face/)

## Acknowledgements

**Contributors**:
  * Ported by [Joey Lee](https://jk-lee.com)

**Credits**:
  * [Face-api.js](https://github.com/justadudewhohacks/face-api.js/blob/master/README.md) is an open source project built by [Vincent Mühler](https://github.com/justadudewhohacks).

## Source Code

* [/src/FaceApi](https://github.com/ml5js/ml5-library/tree/main/src/FaceApi)
