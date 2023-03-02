#if __cplusplus
extern "C" {
#endif

const void* loadModel(const char* modelPath);
void closeModel(const void* model);
float* predictWith(const void* model, float* mel_segment);

#if __cplusplus
}   // Extern C
#endif
