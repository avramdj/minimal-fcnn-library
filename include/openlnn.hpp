#ifndef LIN_NN_LIB_HPP
#define LIN_NN_LIB_HPP

#include <vector>
#include <cmath>

namespace model {
    typedef std::vector<double> array;
    //array flatten();
    
    class Net {
    private:
        class Layer;
        std::vector<Layer *> layers;
        void forward(array& inVals);
        Layer* inLayer;
        Layer* outLayer;
        double recent_error = 0;
        double error_smooth = 100;
    public:
        Net();
        void add_layer(int size);
        void fit(std::vector<array> input_data, std::vector<array> output_data, int num_epochs);
        void backpropagate(array target);
        array predict(array& in);
        array getResult();
        void evaluate(std::vector<array> input_data, std::vector<array> output_data);
        void compile(float trainRate);
    };

    class Net::Layer {
    private:
        class Neuron;
        int size = 0;
    public:
        std::vector<Neuron *> neurons;
        Layer(int size, int prevSize);
        int getSize();
        void set(array& inVals);
        void feed(Layer* prev);
        void calcGradientTarget(array& target);
        void calcGradient(Layer* nextLayer);
        array toArray();
        double sumContrib(int neuronIndex);
        void updateWeights(Layer* prevLayer);
        void compile(float trainRate);
    };

    class Net::Layer::Neuron{
    private:
        double value;
        double gradient = 0;
        double momentum = 0.15;
        double trainRate = 0.1;
    public:
        array inputWeights;
        array inputDeltas;
        Neuron(double v, int inSize);
        void setOut(double v);
        double getVal();
        double getWeight(int i);
        void setGradient(double g);
        double getGradient();
        void updateWeights(Layer* prev);
        void setTrainRate(float trainRate);
    };
}

namespace util{
    double activationDerivative(double x);
    double activation(double x);
    model::array softMaxArr(model::array& arr);
    int softMax(model::array& arr);
}

namespace dataset {
    typedef std::vector<std::vector<double>> Dataset;
    std::vector<Dataset> split(Dataset& inData, Dataset& outData, float ratio);
}

#endif