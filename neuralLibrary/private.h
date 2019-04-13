#pragma once
#include <vector>
#include <iostream>


/*

LAYER
-----

*/

class Layer {
	public:
	struct FloatArray *input;
	struct FloatArray *output;


	virtual void forward();
	virtual struct FloatArray* backpropagate(struct FloatArray* error,
		float learningRate);
	virtual void setParentLayer(Layer &parent);
	virtual void setupAsInputLayer(int sizex, int sizey, int sizez);
	virtual void save(std::ofstream &stream);

	virtual ~Layer();

};

/*

PERCEPTRON LAYER
---------------

*/

struct Neuron {

	std::vector<float> weights;
	float bias;

	Neuron(int inputCount);

	float compute(struct FloatArray &input);
};

class PerceptronLayer : public Layer {

	public:
	int neuronsCount;
	std::vector<Neuron*> neurons;

	PerceptronLayer(int neuronsCount);

	void setParentLayer(Layer &parent);
	void setupAsInputLayer(int sizex, int sizey, int sizez);
	void forward();
	FloatArray* backpropagate(FloatArray *error, float learningRate);
	void save(std::ofstream &stream);

	~PerceptronLayer();
};


/*

CONVOLUTONAL LAYER
----------------

*/

// Convolution operator
struct FloatArray* operator*(struct FloatArray &array1, struct FloatArray &array2);
// Add a float to all floats in an FloatArray
struct FloatArray* operator+(struct FloatArray &array, float x);
// Overlap two float arrays
struct FloatArray* overlap(struct FloatArray &array1, struct FloatArray &array2);


class Filter {
	
	public:
	FloatArray *data;
	float bias;

	Filter();
	~Filter();
};

class ConvolutionalLayer : public Layer {
	
	public:
	int filtersCount;
	std::vector<Filter*> filters;

	ConvolutionalLayer(int filtersCount);
		
	void setParentLayer(Layer &parent);	
	void setupAsInputLayer(int sizex, int sizey, int sizez);
	void forward();
	FloatArray* backpropagate(FloatArray *error, float learningRate);
	void save(std::ofstream &stream);

	~ConvolutionalLayer();
};


/*

ACTIVATION LAYER
----------------

*/

class ActivationLayer : public Layer {

	public:
	float (*activation)(float);
	float (*activationDerivative)(float);
	std::string functionName;

	ActivationLayer(std::string functionName);

	void forward();
	FloatArray* backpropagate(FloatArray *error, float learningRate);
	void setParentLayer(Layer &parent);
	void setupAsInputLayer(int sizex, int sizey, int sizez);
	void save(std::ofstream &stream);

	~ActivationLayer();

};

