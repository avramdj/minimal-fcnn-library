#include "../include/openlnn.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <vector>

int main(){

    //loading dataset
    std::ifstream file("randomXOR.txt");
    std::string str;
    std::vector<std::vector<double>> inData; 
    std::vector<std::vector<double>> outData; 
    while(std::getline(file, str)){
        std::stringstream s(str);
        std::string tmp;
        s >> tmp;
        if(tmp == "in"){
            std::vector<double> data;
            float in;
            while(s >> in){
                data.push_back(in);
            }
            inData.push_back(data);
        } else if (tmp == "out"){
            std::vector<double> data;
            float out;
            while(s >> out){
                data.push_back(out);
            }
            outData.push_back(data);
        }
    }


    std::vector<dataset::Dataset> splitData = dataset::split(inData, outData, 0.8);

    std::vector<std::vector<double>> *trainDataIn, *trainDataOut, *testDataIn, *testDataOut;

    trainDataIn = &splitData[0];
    trainDataOut = &splitData[1];
    testDataIn = &splitData[2];
    testDataOut = &splitData[3];

    int inSize = inData[0].size();
    int outSize = outData[0].size();

    //creating linear fully connected model
    model::Net model;

    model.add_layer(inSize);
    model.add_layer(16);
    model.add_layer(64);
    model.add_layer(64);
    model.add_layer(32);
    model.add_layer(outSize);

    model.compile(0.05);

    model.fit(*trainDataIn, *trainDataOut, 30);

    model.evaluate(*testDataIn, *testDataOut);

    //prediction
    std::cout << std::endl << "Predicting random element:" << std::endl;
    int el = rand() % inData.size();
    auto inSample = inData[el];
    auto outSample = outData[el];
    auto pred = model.predict(inSample);
    for(float d : inSample){
        std::cout << d << " ";
    }
    std::cout << std::endl << "predicted: ";
    for(float p : pred){
        std::cout << p << " ";
    }
    std::cout << std::endl << "target: ";
    for(float o : outSample){
        std::cout << o << " ";
    }
    std::cout << std::endl;

    exit(EXIT_SUCCESS);
}