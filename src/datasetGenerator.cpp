#include <iostream>
#include <fstream>
#include <time.h>

int main(){

    std::ofstream file;
    file.open("randomXOR.txt");
    srand(time(NULL));
    for(int i = 0; i < 4000; i++){
        int a = rand()%4;
        int b = rand()%4;
        int c = rand()%4;
        int d = rand()%4;
        int res = a ^ b ^ c ^ d;
        file << "in ";
        file << (a & 1) << " ";
        file << (a & (2 << 1)) << " ";
        file << (b & 1) << " ";
        file << (b & (2 << 1)) << " ";
        file << (c & 1) << " ";
        file << (c & (2 << 1)) << " ";
        file << (d & 1) << " ";
        file << (d & (2 << 1)) << " ";
        file << std::endl;
        file << "out " << (res & 1) << " " << (res & (2 << 1)) << std::endl;
    }
    file.close();

    return 0;
}