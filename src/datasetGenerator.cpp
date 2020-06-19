#include <iostream>
#include <fstream>
#include <time.h>
#include <vector>
#include <string.h>

void print_input(std::vector<int>& in, std::ofstream& file, bool isInput){
    std::string str;
    if(isInput){
        str += "in ";
    } else {
        str += "out ";
    }
    for(auto& el : in){
        str += std::to_string(el) + " ";
    }
    file << str << std::endl;
}

#define N       3
#define SIZE    1000

int main(){

    std::ofstream file;
    file.open("randomXOR.txt");
    srand(time(NULL));
    for(int i = 0; i < SIZE; i++){
        std::vector<int> input;
        std::vector<int> output;
        for(int i = 0; i < N; i++){
            input.push_back(rand()%2);
        }
        int res = input[0];
        for(int i = 1; i < N; i++){
            res ^= input[i];
        }
        output.push_back(res);
        print_input(input, file, true);
        print_input(output, file, false);
    }
    file.close();

    return 0;
}