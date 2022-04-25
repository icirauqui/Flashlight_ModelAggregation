#include <iostream>

#include "agent.hpp"







int main() {
    std::cout << "Flashlight_ModelAggregation" << std::endl;

    agent* ag1 = new agent();

    ag1->populate("test");

    ag1->train(1);



    return 0;
}