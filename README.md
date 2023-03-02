# callCoreMLFromCpp
example of using CoreML from c++

1. (Done) generate ```encoder_{modelSize}_fp16.mlmodel``` by ```convert_whisper_encoder_to_coreml.py```
2. (Done) model to coreml/CoremlEncoder.mlmodel
2. (Done) generate ```mlmodelc/``` and objc srcs in coreml folder by
```
xcrun coremlc compile CoremlEncoder.mlmodel .
xcrun coremlc generate CoremlEncoder.mlmodel .
```
3. make
4. ./main (it should generate same result from pytorch output)
