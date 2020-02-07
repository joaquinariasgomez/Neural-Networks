#include "Perceptron.hh"
#include <iostream>

Perceptron::Perceptron(std::vector<float> x_, std::vector<float> w_, float target_): output(0), target(target_), x(x_), w(w_) {}

void Perceptron::compute() {
    float z = multiplyVectors(x, w);
    output = stepFunction(z);
    updateWeights();
}

float Perceptron::getOutput() const {return output;}

void Perceptron::printWeights() const {
    for(int i=0; i<w.size(); ++i) {
        std::cout << "w" << i << ": " << w[i] << "\t";
    }
}

float Perceptron::stepFunction(float input) const {
    if(input <= 0) {
        return 0;
    }
    else {
        return 1;
    }
}

void Perceptron::updateWeights() {
    float n = 0.5;  //Learning rate

    for(int i=0; i<w.size(); ++i) {
        w[i] = w[i] + n*(target - output)*x[i];
    }
}

float Perceptron::multiplyVectors(std::vector<float> v1, std::vector<float> v2) const {
    if(v1.size() != v2.size()) return 0;

    float sum = 0;
    for(int i=0; i<v1.size(); ++i) {
        sum += v1[i] * v2[i];
    }
    return sum;
}
