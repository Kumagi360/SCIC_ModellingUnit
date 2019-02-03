Perceptron brain;

Point[] points = new Point[100];

int trainingIndex = 0;



void setup() {
  size(800, 800);
  brain = new Perceptron();


  for (int i = 0; i < points.length; i++) {
    points[i] = new Point();
  }

  //demonstration (prints -1 as computer guess)
  float[] inputs = {-1, 0.5};
  int guess = brain.guess(inputs);
  println(guess);
}



void draw() {
  background(255);
  stroke(0);
  line(0, 0, width, height);
  for (Point pt : points) {
    pt.show();
  }

  //spawns points and assigns inputs
  for (Point pt : points) {
    float[] inputs = {pt.x, pt.y};
    int target = pt.label;
    int guess = brain.guess(inputs);
    if (guess == target) {
      fill(0, 255, 0);
    } else {
      fill(255, 0, 0);
    }
    noStroke();
    ellipse(pt.x, pt.y, 16, 16);
  }
}


//Trains on mousepressed
void mousePressed() {
  for (Point pt : points) {
     Point training = points[trainingIndex];
  float[] inputs = {training.x, training.y};
  int target =training.label;
  brain.train(inputs, target);
  trainingIndex++;
  if (trainingIndex == points.length) {
    trainingIndex = 0;
  }
  }
}

// The activation step function
int sign(float n) {
  if (n >= 0) {
    return 1;
  } else {
    return -1;
  }
}


class Perceptron {
  //intialises weights and learning rate
  float[] weights = new float[2];
  float lr = 0.1;

  // Constructor
  Perceptron() {
    // Initialize the weights randomly
    for (int i = 0; i < weights.length; i++) {
      weights[i] = random(-1, 1);
    }
  }

  //guess function
  int guess(float[] inputs) {
    float sum = 0;
    for (int i = 0; i < weights.length; i++) {
      sum += inputs[i]*weights[i];
    }
    int output = sign(sum);
    return output;
  }

  //train function 
  void train(float[] inputs, int target) {
    int guess = guess(inputs);
    int error = target - guess;
    // Tune all the weights
    for (int i = 0; i < weights.length; i++) {
      weights[i] += error * inputs[i] * lr;
    }
  }
}


//making the colouring of the points correpsond to its cost
class Point {
  float x;
  float y;
  int label;

  Point() {
    x = random(width);
    y = random(height);

    if (x > y) {
      label = 1;
    } else {
      label = -1;
    }
  }

  void show() {
    stroke(0);
    if (label == 1) {
      fill(255);
    } else {
      fill(0);
    }
    ellipse(x, y, 32, 32);
  }
}
