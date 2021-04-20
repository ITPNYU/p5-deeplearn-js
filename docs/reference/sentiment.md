# Sentiment


<center>
    <img style="display:block; max-height:20rem" alt="illustration of hand drawing a heart" src="_media/reference__header-sentiment.png">
</center>


## Description

Sentiment is a model trained to predict the sentiment of any given text. The default model, currently 'moviereviews', is trained using IMDB reviews that have been truncated to a maximum of 200 words, only the 20000 most used words in the reviews are used.

## Quickstart

```js
// Create a new Sentiment method
const sentiment = ml5.sentiment('movieReviews', modelReady);

// When the model is loaded
function modelReady() {
  // model is ready
  console.log('Model Loaded!');
}

// make the prediction
const prediction = sentiment.predict(text);
console.log(prediction);
```


## Usage

### Initialize

```js
const sentiment = ml5.sentiment(model, ?callback);
```

#### Parameters
* **model**: REQUIRED. Defaults to 'moviereviews'. You can also use a path to a `manifest.json` file via a relative or absolute path.
* **callback**: OPTIONAL. A callback function that is called once the model has loaded. If no callback is provided, it will return a promise that will be resolved once the model has loaded.


### Properties

***
#### .ready
> Boolean value that specifies if the model has loaded.
***

***
#### .model
> The model being used.
***


### Methods


***
#### .predict()
> Given a number, will make magicSparkles

```js
sentiment.predict(text);
```

📥 **Inputs**

* **text**: Required. String. A string of text to predict


📤 **Outputs**

* **Object**: Scores the sentiment of given text with a value between 0 ("negative") and 1 ("positive").

***


## Examples

**p5.js**
* [Sentiment_Interactive](https://github.com/ml5js/ml5-library/tree/main/examples/p5js/Sentiment/Sentiment_Interactive)

**p5 web editor**
* [Sentiment_Interactive](https://editor.p5js.org/ml5/sketches/Sentiment_Interactive)

**plain javascript**
* [Sentiment_Interactive](https://github.com/ml5js/ml5-library/tree/main/examples/javascript/Sentiment/Sentiment_Interactive)

## Demo

**plain javascript**
* [Sentiment_Interactive Demo] (http://ml5demo.arifnezami.com/javascript/sentiment/sentiment_interactive/)

## Tutorials

No tutorials yet - contribute one today!


## Model and Data Provenance
> A project started by [Ellen Nickles](https://github.com/ellennickles/)

### Models Overview

A nice description of the models overview

#### Sentiment - Movie reviews - Model Biography

- **Description**
  - This a pre-trained model to detect whether the sentiment of text is positive or negative. 
- **Developer and Year**
  - Google’s TensorFlow.js team. TensorFlow.js, a JavaScript library from TensorFlow, an open source machine learning platform developed by Google.
- **Purpose and Intended Users**
  - From the website: TensorFlow is an open source machine learning platform that “has a comprehensive, flexible ecosystem of tools, libraries, and community resources that lets researchers push the state-of-the-art in ML and developers easily build and deploy ML-powered applications.” This model is available for use in the ml5 library because Tensorflow licenses it with Apache License 2.0.
- **Hosted Location**
  - It is hosted by Google and imported into ml5 from Google’s collection of Tensorflow.js models.
- **ml5 Contributor and Year**
  - Ported by Itay Niv in 2019
- **References**
  - ml5 Contributor [Itay Niv](https://github.com/itayniv)
  - TensorFlow.js Example [Sentiment Analysis](https://github.com/tensorflow/tfjs-examples/tree/482226b15a757f39871038f35b3b8aad7729e594/sentiment)

#### Sentiment - Movie reviews - Data Biography

- **Description**
  - According to the original description on this ml5 reference page, the TensorFlow.js Sentiment Analysis example, the example’s source citation, the training dataset likely consists of 25,000 highly-polarized IMDB reviews. 
- **Source**
  - The TensorFlow.js example cites the data source as Keras, “a deep learning API written in Python, running on top of the machine learning platform TensorFlow” and “developed with a focus on enabling fast experimentation.” Keras in turn cites the source as the Large Movie Review Dataset.
- **Collector and Year**
  - The Large Movie Review Dataset was created at Stanford University by Andrew Maas in 2011.
- **Collection Method**
  - TBD
- **Purpose and Intended Users**
  - The Large Movie Review Dataset was created for research in the area of Computational Linguistics 
- **References**
  - TensorFlow.js Example [Sentiment Analysis](https://github.com/tensorflow/tfjs-examples/tree/482226b15a757f39871038f35b3b8aad7729e594/sentiment)
  - Website [Keras IMDB movie review sentiment classification dataset](https://keras.io/api/datasets/imdb/)
  - Website [Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)



## Acknowledgements

**Contributors**:
  * Itay Niv

**Credits**:
  * Paper Reference | Website URL | Github Repo | Book reference | etc


## Source Code

[/src/Sentiment/](https://github.com/ml5js/ml5-library/tree/main/src/Sentiment)
