main: objcWrapper.o
	c++ -I. main.cpp objcWrapper.o -o main
objcWrapper.o: objcWrapper.h objcWrapper.mm coreml/CoremlEncoder.h coreml/CoremlEncoder.m
	clang -shared -undefined dynamic_lookup -fobjc-arc objcWrapper.mm coreml/CoremlEncoder.m -o objcWrapper.o -framework Foundation -framework CoreML -framework Accelerate
clean:
	rm main objcWrapper.o
