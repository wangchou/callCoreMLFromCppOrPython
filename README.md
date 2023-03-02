# callCoreMLFromCpp
An example of using CoreML from c++ for Whisper Encoder

1. (Done) generate ```encoder_{modelSize}_fp16.mlmodel``` by ```convert_whisper_encoder_to_coreml.py```
2. (Done) move model to coreml/CoremlEncoder.mlmodel
3. (Done) generate ```mlmodelc/``` and its objc srcs in coreml folder by
```
xcrun coremlc compile CoremlEncoder.mlmodel .
xcrun coremlc generate CoremlEncoder.mlmodel .
```
4. ```make```
5. ```./main``` (it should generate same result from pytorch output)

note: currently coreml/ folder contains 1-3 result from Whisper Tiny model
