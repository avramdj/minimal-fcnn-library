#include "../include/libfcnn.hpp"
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <unistd.h>
#include <algorithm>
#include <iomanip>
#include <time.h>
#include <numeric>

model::Net::Net(){
    srand(time(NULL));
};

double model::activationDerivative(double x){
    double sech = 1.0 / std::cosh(x);
    return sech * sech;
}

double model::activation(double x){
    return tanh(x);
}

model::array softMaxArr(model::array& arr){
    model::array sft(arr.size(), 0);
    double m = *(std::max_element(arr.begin(), arr.end()));
    double sum = std::accumulate(arr.begin(), arr.end(), 0);
    const double scale = m + log(sum);
    for (int i = 0;  i < arr.size();  i++) {
        arr[i] = expf(arr[i] - scale);
    }   
    return sft;
}

int maxIdx(model::array& arr){
    auto maxel = std::max_element(arr.begin(), arr.end());
    return int(maxel - arr.begin());
}

void model::Net::add_layer(int size){
    int prevSize = 0;
    if(layers.size() != 0){
        prevSize = layers.back()->getSize();
    }
    outLayer = new Layer(size, prevSize);
    if(layers.size() == 0){
        inLayer = outLayer;
    }
    layers.push_back(outLayer);
}

void model::Net::forward(model::array& inVals){
    inLayer->set(inVals);
    for(int i = 1; i < layers.size(); i++){
        layers[i]->feed(layers[i-1]);
    }
}

/* 
@params: float trainRate (0 - 1.0), int batchSize
 */
void model::Net::compile(float trainRate, int batchSize){
    this->trainRate = trainRate;
    this->batchSize = batchSize;
}

void model::Net::Layer::set(model::array& inVals){
    if(inVals.size() != this->getSize()){
        throw std::invalid_argument("Layer and data don't fit");
    }
    for(int i = 0; i < this->getSize(); i++){
        neurons[i]->setOut(inVals[i]);
    }
}

void model::Net::fit(model::Dataset input_data, model::Dataset output_data, int num_epochs){
    if(input_data.size() != output_data.size()){
        throw std::invalid_argument("Input and output data size must be equal!");
    }
    for(int epoch = 1; epoch <= num_epochs; epoch++){

        std::cout <<  "Epoch: " << epoch << std::string(5 - std::to_string(epoch).length(), ' ' );

        for(int i = 0; i < input_data.size(); i++){
            forward(input_data[i]);
            model::array result = getResult();
            model::array target = output_data[i];

            double error = 0.0;
            for(int i = 0; i < outLayer->getSize(); i++){
                double dlt = pow(target[i] - outLayer->neurons[i]->getVal(), 2);
                error += dlt;
            }
            error /= outLayer->getSize();
            //error = sqrt(error);
            if(i%batchSize == 0){
                backpropagate(target);
            }
            recent_error = (recent_error * error_smooth + error) / (error_smooth + 1.0);
        }
        std::cout << " -- loss " << recent_error << std::endl;
    }
    std::cout << std::endl;
}

void model::Net::evaluate(model::Dataset input_data, model::Dataset output_data){
    if(input_data.size() != output_data.size()){
        throw std::invalid_argument("Input and output data size must be equal!");
    }
    for(int i = 0; i < input_data.size(); i++){
        forward(input_data[i]);
        model::array result = getResult();
        model::array target = output_data[i];

        double error = 0.0;
        for(int i = 0; i < outLayer->getSize(); i++){
            double dlt = pow(target[i] - outLayer->neurons[i]->getVal(), 2);
            error += dlt;
        }
        error /= outLayer->getSize();
        //error = sqrt(error);
        recent_error = (recent_error * error_smooth + error) / (error_smooth + 1.0);
    }
    std::cout << "Train evaluation error: " << recent_error << std::endl;
}

void model::Net::backpropagate(model::array target){

    outLayer->calcGradientTarget(target);
    for(int i = layers.size() - 2; i > 0; i--){
        Layer* currLayer = layers[i];
        Layer* nextLayer = layers[i+1];
        currLayer->calcGradient(nextLayer);
    }
    for(int i = layers.size() - 1; i > 0; i--){
        Layer* currLayer = layers[i];
        Layer* prevLayer = layers[i-1];
        currLayer->updateWeights(prevLayer, trainRate);
    }
}

model::array model::Net::predict(model::array& in){
    model::array v;
    forward(in);
    return getResult();
}

model::array model::Net::getResult(){
    return outLayer->toArray();
}

model::Net::Layer::Layer(int size, int prevSize){
    this->size = size;
    neurons.resize(size);
    for(int i = 0; i < size; i++){
        neurons[i] = new Neuron(0.0, prevSize);
    }
}

int model::Net::Layer::getSize(){
    return this->size;
}

void model::Net::Layer::feed(Layer* prev){
    for(Neuron* neuron : neurons){
        double out = 0;

        for(int i = 0; i < prev->neurons.size(); i++){
            out += prev->neurons[i]->getVal() * neuron->getWeight(i);
        }
        out = model::activation(out);
        neuron->setOut(out);
    }
}

void model::Net::Layer::calcGradient(Layer* next){
    for(int i = 0; i < neurons.size(); i++){
        double grad = next->sumContrib(i) * model::activationDerivative(neurons[i]->getVal());
        neurons[i]->setGradient(grad);
    }
}

void model::Net::Layer::calcGradientTarget(model::array& target){
    for(int i = 0; i < neurons.size(); i++){
        double grad = (target[i] - neurons[i]->getVal()) * model::activationDerivative(neurons[i]->getVal());
        neurons[i]->setGradient(grad);
    }
}

void model::Net::Layer::updateWeights(Layer* prevLayer, float trainRate){
    for(Neuron* neuron : neurons){
        neuron->updateWeights(prevLayer, trainRate);
    }
}

model::array model::Net::Layer::toArray(){
    model::array result(neurons.size());
    for(int i = 0; i < result.size(); i++){
        result[i] = neurons[i]->getVal();
    }
    return result;
}

model::Net::Layer::Neuron::Neuron(double v, int inSize){
    value = v;
    inputWeights.resize(inSize);
    for(double& weight : inputWeights){
        weight = rand()/double(RAND_MAX);
    }
    inputDeltas.resize(inSize, 0);
}

double model::Net::Layer::Neuron::getVal(){
    return value;
}

double model::Net::Layer::Neuron::getWeight(int i){
    return inputWeights[i];
}

void model::Net::Layer::Neuron::setOut(double v){
    value = v;
}

void model::Net::Layer::Neuron::setGradient(double g){
    gradient = g;
}

double  model::Net::Layer::Neuron::getGradient(){
    return gradient;
}

double model::Net::Layer::sumContrib(int neuronIndex){
    double sum = 0;
    for(int i = 0; i < neurons.size(); i++){
        sum += neurons[i]->inputWeights[neuronIndex] * neurons[i]->getGradient();
    }
    return sum;
}

void model::Net::Layer::Neuron::updateWeights(Layer* prev, float trainRate){
    for(int i = 0; i < inputWeights.size(); i++){
        double oldDelta = inputDeltas[i];
        inputDeltas[i] = trainRate * prev->neurons[i]->getVal() * gradient + momentum * oldDelta;
        inputWeights[i] += inputDeltas[i];
    }
}

/* 
Splits input and output data into a trainIn, trainOut, testIn, testOut

Returns: a vector of datasets (of size 4), indices 0, 1, 2, 3 are
trainIn, trainOut, testIn and testOut respectively
 */
std::vector<model::Dataset> model::split(model::Dataset& inData, 
                                            model::Dataset& outData, float ratio){
        if(ratio < 0 || ratio > 1){
            throw std::invalid_argument("Ratio must be between 0 and 1");
        }
        int n = int(inData.size() * ratio);
        std::vector<Dataset> sub_v;
        sub_v.emplace_back(inData.begin(), inData.begin() + n);
        sub_v.emplace_back(outData.begin(), outData.begin() + n);
        sub_v.emplace_back(inData.begin() + n, inData.end());
        sub_v.emplace_back(outData.begin() + n, outData.end());
        return sub_v;
}