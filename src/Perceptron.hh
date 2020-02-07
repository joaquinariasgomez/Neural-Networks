#include <vector>
#pragma once

class Perceptron {
    public:
        Perceptron(std::vector<float> x_, std::vector<float> w_);
        void compute(); //virtual
        float getOutput() const;
    protected:
        float stepFunction(float input) const;
        float multiplyVectors(std::vector<float> v1, std::vector<float> v2) const;

        float output;
        std::vector<float> x;
        std::vector<float> w;
};