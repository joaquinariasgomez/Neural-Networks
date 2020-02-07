#include <iostream>
#include "Perceptron.hh"
#include "BiasNeuron.hh"

int main() {
    /*std::vector<float> x{1,2,3};
    std::vector<float> w{1,1,1};
    float target = 0;
    Perceptron p(x, w, target);
    for(int i=0; i<10; ++i) {
        p.printWeights();
        p.compute();
        std::cout << "\tOutput: " << p.getOutput() << std::endl;
    }*/
    //Lets try to build a multilayer Perceptron!
    float x1 = 1;
    float x2 = 1;
    std::vector<float> x00{1, x1, x2};
    std::vector<float> x01{1, x1, x2};
    std::vector<float> w00{-1.5, 1, 1};
    std::vector<float> w01{-0.5, 1, 1};
    Perceptron p00(x00, w00, 0);
    Perceptron p01(x01, w01, 0);
    p00.compute();
    p01.compute();
    std::vector<float> x10{1, p00.getOutput(), p01.getOutput()};
    std::vector<float> w10{-0.5, -1, 1};
    Perceptron p10(x10, w10, 1);
    p10.compute();
    for(int i=0; i<10; ++i) {
        std::cout << "\tOutput: " << p10.getOutput() << std::endl;
        p10.compute();
    }
}