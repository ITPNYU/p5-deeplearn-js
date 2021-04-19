# Object Detector


<center>
    <img style="display:block; max-height:20rem" alt="cat detected by yolo" src="_media/reference__header-yolo.png">
</center>


## Description

Real-time object detection system using either [YOLO](https://pjreddie.com/darknet/yolo/) or [CocoSsd](https://github.com/tensorflow/tfjs-models/tree/master/coco-ssd) model.

## Quickstart

```js
const video = document.getElementById('video');

// Create a ObjectDetector method
const objectDetector = ml5.objectDetector('cocossd', {}, modelLoaded);

// When the model is loaded
function modelLoaded() {
  console.log('Model Loaded!');
}

// Detect objects in the video element
objectDetector.detect(video, (err, results) => {
  console.log(results); // Will output bounding boxes of detected objects
});
```


## Usage

### Initialize

```js
const objectDetector = ml5.objectDetector(modelNameOrUrl);
// OR
const objectDetector = ml5.objectDetector(modelNameOrUrl, ?options, ?callback);
```

#### Parameters
* **modelNameOrUrl**: A String value of a valid model OR a url to a `model.json` that contains a pre-trained model. Models available are: ['cocossd'](https://github.com/tensorflow/tfjs-models/tree/master/coco-ssd), ['yolo'](https://pjreddie.com/darknet/yolo/)
* **options**: Optional. An object describing a model accuracy and performance. For YOLO this are: `{ filterBoxesThreshold: 0.01, IOUThreshold: 0.4, classProbThreshold: 0.4 }`
* **callback**: Optional. A function to run once the model has been loaded. If no callback is provided, it will return a promise that will be resolved once the model has loaded.

### Methods


***
#### .detect()
> Given an image or video, returns an array of objects containing class names, bounding boxes and probabilities.

```js
objectDetector.detect(input, ?callback);
```

📥 **Inputs**

* **input**: A HTML video or image element or a p5 image or video element. If no input is provided, the default is to use the video given in the constructor.
* **callback**: A function to run once the model has made the prediction. If no callback is provided, it will return a promise that will be resolved once the model has made a prediction.

📤 **Outputs**

* **Object**: returns an array of objects containing class names, bounding boxes and probabilities.

***


## Examples


**p5.js**
* [COCOSSD_Video](https://github.com/ml5js/ml5-library/tree/main/examples/p5js/ObjectDetector/ObjectDetector_COCOSSD_Video)
* [COCOSSD_single_image](https://github.com/ml5js/ml5-library/tree/main/examples/p5js/ObjectDetector/ObjectDetector_COCOSSD_single_image)

**p5 web editor**

* [COCOSSD_Video](https://editor.p5js.org/ml5/sketches/ObjectDetector_COCOSSD_Video)
* [COCOSSD_single_image](https://editor.p5js.org/ml5/sketches/ObjectDetector_COCOSSD_single_image)
* [YOLO_single_image](https://editor.p5js.org/ml5/sketches/ObjectDetector_YOLO_single_image)
* [YOLO_webcam](https://editor.p5js.org/ml5/sketches/ObjectDetector_YOLO_webcam)

**plain javascript**
* [COCOSSD_single_image](https://github.com/ml5js/ml5-library/tree/main/examples/javascript/ObjectDetector/COCOSSD_single_image)
* [COCOSSD_webcam](https://github.com/ml5js/ml5-library/tree/main/examples/javascript/ObjectDetector/COCOSSD_webcam)
* [YOLO_single_image](https://github.com/ml5js/ml5-library/tree/main/examples/javascript/ObjectDetector/YOLO_single_image)
* [YOLO_webcam](https://github.com/ml5js/ml5-library/tree/main/examples/javascript/ObjectDetector/YOLO_webcam)

## Demo

No demos yet - contribute one today!

## Tutorials

### ml5.js: Object Detection (Coding Train)
<iframe width="560" height="315" src="https://www.youtube.com/embed/QEzRxnuaZCk"  frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Model and Data Provenance
> A project started by [Ellen Nickles](https://github.com/ellennickles/)

### Models Overview

The ml5 Object Detection method takes image classification one step further in a couple of ways. It identifies multiple objects as well as their location in images or video by drawing bounding boxes around the detected content. There are two pre-trained model options to use with this ml5’s object detection method: YOLO and COCO-SSD.

Each model is a type of convolutional neural network (CNN). A CNN finds patterns in the pixels of images (e.g. horizontal or vertical edges), and through successive layers of computation finds sets of patterns to identify more complex patterns (e.g. corners or circles), eventually detecting intricate patterns that it predicts belong to a particular category (e.g. dog or airplane). The categories depend on how images in the model’s training dataset are labeled.


#### COCO-SSD - Model Biography

- **Description**
  - This model detects objects defined in the COCO dataset, which is a large-scale object detection, segmentation, and captioning dataset.
- **Developer and Year**
  - This pre-trained model was developed by the TensorFlow.js team in 2018, where it is currently maintained. TensorFlow.js, a JavaScript library from TensorFlow.
- **Purpose and Intended Users**
  - From the website: TensorFlow is an open source machine learning platform that “has a comprehensive, flexible ecosystem of tools, libraries, and community resources that lets researchers push the state-of-the-art in ML and developers easily build and deploy ML-powered applications.” This model is available for use in the ml5 library because Tensorflow licenses it with Apache License 2.0.
- **Hosted Location**
  - As of June 2019, ml5 imports COCO-SSD from TensorFlow’s models on the NPM database. This means that your ml5 sketch will automatically use the most recent version distributed on NPM.
- **ml5 Contributor and Year**
  - [Tirta Rachman](https://www.tirtawr.com/software-projects) 2019
- **References**
  - ml5 Contributor [Tirta Wening Rachman](https://github.com/tirtawr)
  - Website [TensorFlow](https://www.tensorflow.org/)
  - NPM Readme [@tensorflow-models/coco-ssd](https://www.npmjs.com/package/@tensorflow-models/coco-ssd)
  - GitHub Repository [TensorFlow tfjs-models Object Detection (coco-ssd)](https://github.com/tensorflow/tfjs-models/tree/master/coco-ssd)

#### COCO-SSD - Data Biography

- **Description**
  - This model uses 80 categories from the COCO image dataset to detect objects. Note that there are some slight differences in category names for this model compared to those used for ml5’s YOLO implementation described above.
- **Source**
  - Same as above
- **Collector and Year**
  - Same as above
- **Collection Method**
  - Same as above
- **Purpose and Intended Users**
  - Same as above
- **References**
  - Website [TensorFlow](https://www.tensorflow.org/)
  - GitHub Repository [TensorFlow tfjs-models Object Detection (coco-ssd)](https://github.com/tensorflow/tfjs-models/tree/master/coco-ssd)
  - Paper [Microsoft COCO: Common Objects in Context](https://arxiv.org/abs/1405.0312)
  - Website [Microsoft COCO: Common Objects in Context](http://cocodataset.org/#home)

#### YOLO - Model Biography

- **Description**
  - YOLO describes a type of neural network architecture used for computer vision and pattern recognition.
- **Developer and Year**
  - YOLO development is led by Joseph Redmon. The ml5 version was contributed by Cristóbal Valenzuela in 2018 and according to the code comments, is heavily derived from Mike Shi’s 2018 TensorFlow.js implementation of a Tiny YOLO model on ModelDepot. ModelDepot is a platform to discuss and share pre-trained machine learning models for a community of researchers and engineers. Tiny YOLO, also developed by Redmon, is smaller and less computationally-intensive than full YOLO models.
  - Redmon has developed three versions of YOLO in recent years. Contributions welcome to confirm which version the ml5 implementation uses. A comment in Shi’s initial commit credits Allen Zelener’s 2017 YAD2K model, which is described as an implementation of YOLO_v2. Redmon provides several open source licenses, including MIT License.
- **Purpose and Intended Users**
  - TBD
- **Hosted Location**
  - Hosted by ml5
- **ml5 Contributor and Year**
  - Ported by Cristóbal Valenzuela in 2018 
- **References**
  - Developer [Joseph Redmon](https://pjreddie.com/)
  - Developer [Mike Shi](https://github.com/MikeShi42)
  - ml5 Contributor [Cristóbal Valenzuela](https://cvalenzuelab.com/)
  - GitHub Repository [Mike Shi’s ModelDepot tfjs-yolo-tiny](https://github.com/ModelDepot/tfjs-yolo-tiny)
  - GitHub Repository [Allan Zelener’s YAD2K: Yet Another Darknet 2 Keras](https://github.com/allanzelener/YAD2K)
  - Paper [YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242)
  - Website [YOLO: Real-Time Object Detection (v2)](https://pjreddie.com/darknet/yolov2/)

#### YOLO - Data Biography

- **Description**
  - This model uses 80 categories from the COCO image dataset to detect objects, indicating that it was likely trained on that dataset.
  - If this model is indeed an implementation of YOLO_v2 (mentioned above), then the paper indicates that it was also trained on images from the ImageNet database.)
- **Source**
  - From the website: the COCO dataset is managed by a number of collaborators from both academic and commercial organizations for “large-scale object detection, segmentation, and captioning,” and according to the paper, images were collected from Flickr. 
- **Collector and Year**
  - The COCO database began in 2014.
- **Collection Method**
  - COCO methods for collecting images and annotating pixels into segments are described  in the paper.
- **Purpose and Intended Users**
  - The COCO dataset was created to advance computer vision research.
- **References**
  - Paper [Microsoft COCO: Common Objects in Context](https://arxiv.org/abs/1405.0312)
  - Website [Microsoft COCO: Common Objects in Context](http://cocodataset.org/#home)



## Acknowledgements

**Contributors**:
  * Cristobal Valenzuela
  * Tirta Rachman
  * Joey Lee

**Credits**:
  * Paper Reference | Website URL | Github Repo | Book reference | etc

## Source Code

* [/src/ObjectDetector](https://github.com/ml5js/ml5-library/tree/main/src/ObjectDetector)
