#include "objcWrapper.h"
#include <stdlib.h>
#include <iostream>
using namespace std;

int main() {
    const void* coremlEncoder = loadModel("coreml/CoremlEncoder.mlmodelc");

    // prepare easy test input like torch.ones
    float* mel = (float *)malloc(sizeof(float) * 80 * 3000);
    for(int i=0;i<80*3000; i++) {
        mel[i] = 1.0;
    }

    // alloc output buffer
    int n_state = 384; // tiny=384
    float* outFloats = (float *)malloc(sizeof(float) * 1500 * n_state);

    predictWith(coremlEncoder, mel, outFloats);

    // it should match
    // pytorch output: {'output': array([[[-0.28637695, -0.25561523, ..., -0.10253906]]], dtype=float32)
    cout << outFloats[0] << " " << outFloats[1] << " " << outFloats[n_state*1500-1];
    closeModel(coremlEncoder);
}
