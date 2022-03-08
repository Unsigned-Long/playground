#include "handler.h"

int main(int argc, char const *argv[])
{
    // load data
    ns_test::Camera camera("../data/mapping");
    // do the process
    camera.process();
    // get the params
    auto inner = camera.innerParamsMatrix();
    auto k_tuple = camera.distortionParams();
    
    return 0;
}
