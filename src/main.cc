#include <iostream>
#include "Perceptron.hh"
#include "BiasNeuron.hh"

int main() {
    std::cout << "Hello world!" << std::endl;
    std::vector<float> x{1,2,3};
    std::vector<float> w{1,1,1};
    Perceptron p(x, w);
    p.compute();
    std::cout << "Output: " << p.getOutput() << std::endl;
}