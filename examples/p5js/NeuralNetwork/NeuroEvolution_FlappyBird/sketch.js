// Daniel Shiffman
// Neuro-Evolution Flappy Bird with ml5.js

const TOTAL = 100;
// Current birds
const birds = [];
// Save any birds that die
let savedBirds = [];
let pipes = [];
let counter = 0;
let slider;

function setup() {
  createCanvas(640, 480);
  // Improves performance for small neural networks and classifySync()
  ml5.tf.setBackend("cpu");

  // Slider for speeding up simulation
  slider = createSlider(1, 10, 1);

  // Create initial population of birds
  for (let i = 0; i < TOTAL; i += 1) {
    birds[i] = new Bird();
  }
}

function draw() {
  // Speed up simulation
  for (let n = 0; n < slider.value(); n += 1) {
    // new pipes every N frames
    if (counter % 75 === 0) {
      pipes.push(new Pipe());
    }
    counter += 1;

    // Run game
    for (let i = pipes.length - 1; i >= 0; i -= 1) {
      pipes[i].update();

      for (let j = birds.length - 1; j >= 0; j -= 1) {
        if (pipes[i].hits(birds[j])) {
          // Save bird if it dies
          savedBirds.push(birds.splice(j, 1)[0]);
        }
      }

      // Remove pipes when they leave
      if (pipes[i].offscreen()) {
        pipes.splice(i, 1);
      }
    }

    // Remove / save any birds that go offscreen
    for (let i = birds.length - 1; i >= 0; i -= 1) {
      if (birds[i].offScreen()) {
        savedBirds.push(birds.splice(i, 1)[0]);
      }
    }

    // Run all birds
    for (const bird of birds) {
      bird.think(pipes);
      bird.update();
    }

    // If all the birds have died go to the next generation
    if (birds.length === 0) {
      counter = 0;
      nextGeneration();
      pipes = [];
    }
  }

  // All the drawing stuff
  background(0);

  for (const bird of birds) {
    bird.show();
  }

  for (const pipe of pipes) {
    pipe.show();
  }
}
