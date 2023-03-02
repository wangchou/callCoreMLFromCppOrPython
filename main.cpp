#include "objcWrapper.h"
#include <stdlib.h>
#include <iostream>
using namespace std;

int main() {
    const void* coremlEncoder = loadModel("coreml/CoremlEncoder.mlmodelc");

    // prepare easy test input like torch.ones
    float* inFloats = (float *)malloc(sizeof(float) * 80 * 3000);
    for(int i=0;i<80*3000; i++) {
        inFloats[i] = 1.0;
    }

    float* outFloats = predictWith(coremlEncoder, inFloats);

    // it should match
    // pytorch output: {'output': array([[[-0.28637695, -0.25561523, ..., -0.10253906]]], dtype=float32)
    cout << outFloats[0] << " " << outFloats[1] << " " << outFloats[384*1500-1];
    closeModel(coremlEncoder);
}
