#include <vector>
#pragma once

class Perceptron {
    public:
        Perceptron(std::vector<float> x_, std::vector<float> w_, float target_);
        void compute(); //virtual
        float getOutput() const;
        void printWeights() const;
    protected:
        float stepFunction(float input) const;
        void updateWeights();
        float multiplyVectors(std::vector<float> v1, std::vector<float> v2) const;

        float output;
        float target;
        std::vector<float> x;
        std::vector<float> w;
};