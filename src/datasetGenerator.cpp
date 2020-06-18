#include <iostream>
#include <fstream>
#include <time.h>

int main(){

    std::ofstream file;
    file.open("randomXOR.txt");
    srand(time(NULL));
    for(int i = 0; i < 4000; i++){
        int a = rand()%2;
        int b = rand()%2;
        int c = rand()%2;
        int d = rand()%2;
        int res = a ^ b ^ c ^ d;
        file << "in ";
        file << (a ) << " ";
        file << (b ) << " ";
        file << (c ) << " ";
        file << (d ) << " ";
        file << std::endl;
        file << "out " << (res)<< std::endl;
    }
    file.close();

    return 0;
}