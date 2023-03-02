#import <CoreML/CoreML.h>
#import <Accelerate/Accelerate.h>
#import "objcWrapper.h"
#import "coreml/CoremlEncoder.h"
#include <stdlib.h>

#if __cplusplus
extern "C" {
#endif

const void* loadModel(const char* modelPath) {
    NSString* modelPathStr = [[NSString alloc] initWithUTF8String:modelPath];
    NSURL* modelURL = [NSURL fileURLWithPath: modelPathStr];

    const void* model = CFBridgingRetain([[CoremlEncoder alloc] initWithContentsOfURL:modelURL error:nil]);
    return model;
}

float* _outFloatArray = (float *)malloc(sizeof(float) * 384 * 1500);

float* predictWith(const void* model, float* mel_segment) {
    MLMultiArray *inMultiArray = [[MLMultiArray alloc] initWithDataPointer: mel_segment
                                                                      shape: @[@1, @80, @3000]
                                                                   dataType: MLMultiArrayDataTypeFloat32
                                                                    strides: @[@(240000), @(3000), @1]
                                                                deallocator: nil
                                                                      error: nil];

    CoremlEncoderOutput *modelOutput = [(id)model predictionFromMel_segment:inMultiArray error:nil];
    MLMultiArray *outMA = modelOutput.output;

    cblas_scopy((int)outMA.count,
                (float*)outMA.dataPointer, 1,
                _outFloatArray, 1);

    return _outFloatArray;
}

void closeModel(const void* model) {
    CFRelease(model);
}

#if __cplusplus
} //Extern C
#endif
