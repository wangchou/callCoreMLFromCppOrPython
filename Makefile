main: objcWrapper.o coreml.o
	c++ -I. main.cpp objcWrapper.o coreml.o -o main -framework Foundation -framework CoreML -framework Accelerate
objcWrapper.o: objcWrapper.h objcWrapper.mm coreml/CoremlEncoder.h coreml.o
	clang++ -I. -c -fobjc-arc objcWrapper.mm -o objcWrapper.o
coreml.o: coreml/CoremlEncoder.h coreml/CoremlEncoder.m
	clang++ -I. -c -fobjc-arc coreml/CoremlEncoder.m -o coreml.o
clean:
	rm main objcWrapper.o coreml.o
