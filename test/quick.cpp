//#include <string>
//#include <cstdio>
//#include <memory>
//#include <stdexcept>
//#include <array>

#include <iostream>
#include <puffinn.hpp>
#include <utils.hpp>

int main() {
    std::cout << "br\n";
    struct utils::Timer t;
    t.start();
    // Code to be timed here
    puffinn::Dataset<puffinn::UnitVectorFormat> train(0,0);
    utils::load(train, "train");
    float execution_time = t.duration();
    std::cout << execution_time << " seconds" << std::endl;
    
}
