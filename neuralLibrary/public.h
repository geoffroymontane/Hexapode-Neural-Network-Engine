#pragma once
#include <vector>
#include <string>

/*
 
ACTIVATION FUNCTIONS
-------------------

*/

float relu(float x);
float reluD(float x);
float sigmoid(float x);
float sigmoidD(float x);



struct FloatArray {
	std::vector<float> array;

	int sizex;
	int sizey;
	int sizez;

	int size;

	FloatArray(int sizex, int sizey, int sizez);

	void set(int x, int y, int z, float f);
	float operator()(int x, int y, int z);
	void apply(float (*funct)(float));
	void print();
};


/*

ERROR FUNCTIONS
--------------

*/

float quadraticError(FloatArray &output, FloatArray &ideal_output);
FloatArray *quadraticErrorGradient(FloatArray &output, FloatArray &ideal_output);

#include "private.h"


/*
 
NEURAL NETWORK
-------------

*/

class Network {

	public:
	std::vector<Layer*> hiddenLayers;
	Layer *inputLayer;
	Layer *outputLayer;

	int inputSizex;
	int inputSizey;
	int inputSizez;

	Network(int inputSizex, int inputSizey, int inputSizez);

	void setInputLayer(Layer* layer);
	void setOutputLayer(Layer* layer);
	
	void addPerceptronLayer(int neuronsCount);
	void addConvolutionalLayer(int filtersCount);
	void addActivationLayer(std::string functionName);

	void build();
	struct FloatArray forward(struct FloatArray &input);

	// error will be freed
	void backpropagate(struct FloatArray *error, float learningRate);

	void save(std::string path);
	static Network retrieve(std::string path);

	~Network();
};

