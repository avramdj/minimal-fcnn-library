#include <iostream>
#include <fstream>
#include <time.h>

int main(){

    std::ofstream file;
    file.open("randomXOR.txt");
    srand(time(NULL));
    for(int i = 0; i < 2000; i++){
        int a = rand() > (RAND_MAX / 2);
        int b = rand() > (RAND_MAX / 2);
        int c = rand() > (RAND_MAX / 2);
        int d = rand() > (RAND_MAX / 2);
        int e = rand() > (RAND_MAX / 2);
        int f = rand() > (RAND_MAX / 2);
        int g = rand() > (RAND_MAX / 2);
        int res = a ^ b ^ c ^ d ^ e ^ f ^ g;
        file << "in " << a << " " << b << " " << c << " " << d << " " << e
        << " " << f << " " << g << std::endl;
        file << "out " << res << std::endl;
    }
    file.close();

    return 0;
}