#include <cmath>
#include <vector>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <thread>

#include "public.h"

using namespace std;


/*
 
ACTIVATION FUNCTIONS
-------------------

*/

float relu(float x) {
	if (x > 0) {
		return x;
	}
	return 0;
}

float reluD(float x) {
	if (x > 0) {
		return 1;
	}
	return 0;
}

float sigmoid(float x) {
	return 1.0 / (exp(-x) + 1.0);
}

float sigmoidD(float x) {
	return exp(-x) / ((exp(-x) + 1.0) * (exp(-x) + 1.0));
}

void Layer::forward(){cout << "err" <<endl;}
void Layer::save(ofstream &stream){return;}
struct FloatArray* Layer::backpropagate(struct FloatArray *error, float
learningRate){return NULL;cout << "err" <<endl;}
void Layer::setParentLayer(Layer &parent){cout << "err" <<endl;}
void Layer::setupAsInputLayer(int sizex, int sizey, int sizez){cout << "err" <<endl;}
Layer::~Layer(){}

/*

ERROR FUNCTIONS
--------------

*/

float quadraticError(FloatArray &output, FloatArray &ideal_output)
{
	float sum = 0;
	for(int i = 0; i < output.size; i++) {
		sum += pow(output.array[i] - ideal_output.array[i], 2);
	}

	return sum;
}

FloatArray *quadraticErrorGradient(FloatArray &output, FloatArray &ideal_output)
{
	FloatArray *gradient=new FloatArray(output.sizex, output.sizey, output.sizez);
	for(int i = 0; i < output.size; i++) {
		gradient->array[i] = 2 *
			(output.array[i] - ideal_output.array[i]);
	}

	return gradient;	
}

/*

PERCEPTRON LAYER
---------------

*/

Neuron::Neuron(int inputCount)
{
	for (int i = 0; i < inputCount; i++){
		float rnd = rand()/(RAND_MAX + 1.) - 0.5;
		weights.push_back(rnd);
	}
	bias = rand()/(RAND_MAX + 1.) - 0.5;
}

float Neuron::compute(struct FloatArray &input)
{
	float output = bias;
	for (int i = 0; i < input.size; i++){
		output += weights[i]*(input.array[i]);
	}

	return output;	
}

PerceptronLayer::PerceptronLayer(int neuronsCount)
{
	this->neuronsCount = neuronsCount;
	output = new FloatArray(neuronsCount, 1, 1);
}

void PerceptronLayer::setParentLayer(Layer &parent)
{
	input = parent.output;
	
	for (int i = 0; i < neuronsCount; i++){
		neurons.push_back(new Neuron(input->size));
	}
}	

void PerceptronLayer::setupAsInputLayer(int sizex, int sizey, int sizez)
{
	input = new FloatArray(sizex, sizey, sizez);

	for (int i = 0; i < neuronsCount; i++){
		neurons.push_back(new Neuron(input->size));
	}
}

void PerceptronLayer::forward()
{
	for (int i = 0; i < neurons.size(); i++){
		output->array[i] = neurons[i]->compute(*input);
	}
}

void PerceptronLayer::save(ofstream &stream)
{
	stream << "layer perceptron" << endl;
	for (int n = 0; n < neurons.size(); n++) {
		stream << "neuron ";
		stream << neurons[n]->bias << " ";
		for (int k = 0; k < neurons[n]->weights.size(); k++) {
			stream << neurons[n]->weights[k] << " ";
		}
		stream << endl;
	}
}

struct FloatArray* PerceptronLayer::backpropagate(FloatArray *error,
	float learningRate)
{

	// Compute parent layer neurons error
	//ERROR
	FloatArray* parentError = new FloatArray(input->sizex, input->sizey, input->sizez);

	for (int i = 0; i < input->size; i++){
		float _parentError = 0;

		for (int n = 0; n < neurons.size(); n++){
			float _error = (*error)(n, 0, 0);
			float weight = neurons[n]->weights[i];

			_parentError += (_error * weight);
			
		}

		parentError->array[i] = _parentError;
	}

	// Update weights
	for (int i = 0; i < neurons.size(); i++){
		float _error = (*error)(i, 0, 0);	

		for (int n = 0; n < input->size; n++){
			float input_ = input->array[n];
			neurons[i]->weights[n] -=
				learningRate * input_ * _error;
		}
		
		// Update bias
		neurons[i]->bias -= learningRate * _error;
	}	

	return parentError;
}

PerceptronLayer::~PerceptronLayer()
{

	for (int i = 0; i < neurons.size(); i++){
		delete neurons[i];
		neurons[i] = 0;
	}

	delete output;
	output = 0;
}


/*

FLOAT 3D ARRAY
-------------

*/

FloatArray::FloatArray(int sizex, int sizey, int sizez) {
	this->sizex = sizex;
	this->sizey = sizey;
	this->sizez = sizez;

	size = sizex * sizey * sizez;
	for(int i = 0; i < size; i++) {
		array.push_back(0);
	}
}

// SET
void FloatArray::set(int x, int y, int z, float f) {
	array[z * sizey * sizex + y * sizex + x] = f;
}

// GET
float FloatArray::operator()(int x, int y, int z)
{
	return array[z * sizey * sizex + y * sizex + x];
}

// Apply function to floats
void FloatArray::apply(float (*funct)(float)) {
	for (int i=0; i < sizex * sizey * sizez; i++) {
		array[i] = funct(array[i]);
	}
}

void FloatArray::print() {
	cout << "FloatArray" << endl << "-----" << endl;
	for (int z = 0; z<sizez; z++) {
		for (int y = 0; y<sizey; y++) {
			for (int x = 0; x<sizex; x++) {
				cout << (*this)(x, y, z) << " ";	
			}
			cout << endl;
		}
		cout << "-----" << endl;
	}
}

// Convolution operator
struct FloatArray* operator*(struct FloatArray &array1, struct FloatArray &array2) {

	struct FloatArray *output = new FloatArray(array1.sizex - array2.sizex + 1, 
				array1.sizey - array2.sizey + 1, 
				array1.sizez - array2.sizez + 1);
	
	for (int z = 0; z<output->sizez; z++) {
		for (int x = 0; x<output->sizex; x++) {
			for (int y = 0; y<output->sizey; y++) {
				float s = 0;
				for (int k = 0; k<array2.sizez; k++) {
					for (int i = 0; i<array2.sizex; i++) {
						for (int j = 0; j<array2.sizey; j++) {
							s += array1(i+x, j+y, k+z) * array2(i, j, k);
						}
					}
				}
				output->set(x, y, z, s);
			}
		}
	}
	
	return output;
}

// Add a float to all floats in an FloatArray
struct FloatArray* operator+(struct FloatArray &array, float x) {
	struct FloatArray* output = new FloatArray(array.sizex, array.sizey, array.sizez);
	for (int i = 0; i < array.sizex * array.sizey * array.sizez; i++) {
		output->array[i] = array.array[i] + x;
	}
	
	return output;
}

// Overlap two float arrays
struct FloatArray* overlap(struct FloatArray &array1, struct FloatArray &array2)
{
	if (array1.sizex == array2.sizex && array1.sizey == array2.sizey) {
		
		FloatArray *output = new FloatArray(array1.sizex, array1.sizey,
						array1.sizez + array2.sizez);
		
		for (int z = 0; z<array1.sizez; z++) {
			for (int x = 0; x<output->sizex; x++) {
				for (int y = 0; y<output->sizey; y++) {
					output->set(x, y, z, array1(x, y, z));
				}
			}
		}
		
		for (int z = array1.sizez; z<output->sizez; z++) {
			for (int x = 0; x<output->sizex; x++) {
				for (int y = 0; y<output->sizey; y++) {
					output->set(x, y, z, array2(x, y,
							z-array1.sizez));
				}
			}
		}

		return output;
	}
	cout << "Error : function overlap. Size mismatch." << endl;
	exit(1);
}

/*

FILTER
-----

*/

Filter::Filter()
{
	bias = rand()/(RAND_MAX + 1.) - 0.5;
	data = new FloatArray(5, 5, 3);
	
	for (int i = 0; i < data->size; i++) {
		data->array[i] = rand()/(RAND_MAX + 1.) - 0.5;
	}
}

Filter::~Filter()
{
	delete data;
}

/*

CONVOLUTONAL LAYER
----------------

*/

ConvolutionalLayer::ConvolutionalLayer(int filtersCount)
{
	for (int i = 0; i < filtersCount; i++) {
		filters.push_back(new Filter());
	}
}

void ConvolutionalLayer::setParentLayer(Layer &parent)
{
	input = parent.output;	
	output = new FloatArray(input->sizex - 5 + 1, 
		input->sizey - 5 + 1, input->sizez - 5 + 1);
}

void ConvolutionalLayer::setupAsInputLayer(int sizex, int sizey, int sizez)
{
	input = new FloatArray(sizex, sizey, sizez);
	output = new FloatArray(sizex - 5 + 1, sizey - 5 + 1, sizez - 5 + 1);
}

void ConvolutionalLayer::forward()
{
	FloatArray *buffer = new FloatArray(output->sizex, output->sizey, output->sizez); 
	FloatArray *newbuffer;
	FloatArray *tmp;	

	for (int i = 0; i < filters.size(); i++) {
		tmp = (*input) * (*filters[i]->data);
		newbuffer = overlap(*buffer, *tmp);
		delete buffer;
		buffer = newbuffer;
		delete tmp;
	}
	
	*output = *buffer;
	delete buffer;
}

FloatArray* ConvolutionalLayer::backpropagate(FloatArray *error,
	float learningRate)
{	

	// Compute parent error
	FloatArray *parentError = new FloatArray(input->sizex, input->sizey,
		input->sizez);

	for (int x = 0; x < input->sizex; x++) { 
	for (int y = 0; y < input->sizey; y++) { 
	for (int z = 0; z < input->sizez; z++) { 
		for (int n = 0; n < filters.size(); n++) {
		for (int i = 0; i < min(input->sizex - 1, x); i++) { 
		for (int j = 0; i < min(input->sizey - 1, y); j++) { 
				parentError->set(x, y, z, (*parentError)(x, y, z) +
					(*filters[n]->data)(i, j, z) *
					(*error)(x - i, y - j, n));
		}
		}
		}
	}
	}
	}

	// Update weights
	for (int n = 0; n < filters.size(); n++) {
		for (int i = 0; i < 5; i++) {  
		for (int j = 0; j < 5; j++) {  
		for (int k = 0; k < 3; k++) {  
			for (int x = 0; x < input->sizex - i; x++) { 
			for (int y = 0; y < input->sizey - j; y++) { 
				filters[n]->data->set(i, j, k,
					(*filters[n]->data)(i, j, k)
					- learningRate * (*error)(x, y, n)
					* (*output)(x + i, x + j, k));
			}
			}
		}
		}
		}
	}
	
	return parentError;	
}

void ConvolutionalLayer::save(ofstream &stream)
{
	stream << "layer convolutional" << endl;
	for (int n = 0; n < filters.size(); n++) {
		stream << "neuron ";
		stream << filters[n]->bias << " ";
		for (int k = 0; k < filters[n]->data->size; k++) {
			stream << filters[n]->data->array[k] << " ";
		}
		stream << endl;
	}
}

ConvolutionalLayer::~ConvolutionalLayer()
{
	delete output;
	for (int i = 0; i < filters.size(); i++) {
		delete filters[i];
	}
}

/*

ACTIVATION LAYER
----------------

*/

ActivationLayer::ActivationLayer(string functionName)
{
	if (functionName == "sigmoid") {
		this->activation = sigmoid;
		this->activationDerivative = sigmoidD;
	}
	else if (functionName == "relu") {
		this->activation = relu;
		this->activationDerivative = reluD;
	}

	this->functionName = functionName;
}

void ActivationLayer::setParentLayer(Layer &parent)
{
	input = parent.output;
	output = new FloatArray(input->sizex, input->sizey, input->sizez);
}

void ActivationLayer::setupAsInputLayer(int sizex, int sizey, int sizez)
{
	input = new FloatArray(sizex, sizey, sizez);
	output = new FloatArray(sizex, sizey, sizez);
}

void ActivationLayer::forward()
{
	for (int i = 0; i < input->size; i++) {
		output->array[i] = activation(input->array[i]);
	}
}

void ActivationLayer::save(ofstream &stream)
{
	stream << "layer activation " << functionName << endl;
}

struct FloatArray* ActivationLayer::backpropagate(FloatArray *error, 
	float learningRate)
{

	FloatArray *newError = new FloatArray(output->sizex, output->sizey,
output->sizez);

	for (int i = 0; i < error->size; i++) {
		newError->array[i] = activationDerivative(input->array[i])
			* error->array[i];
	}

	return newError;
}

ActivationLayer::~ActivationLayer()
{
	delete output;
	output = 0;
}


/*

NEURAL NETWORK
-------------

*/

Network::Network(int inputSizex, int inputSizey, int inputSizez)
{
	srand(time(NULL));
	this->inputSizex = inputSizex;
	this->inputSizey = inputSizey;
	this->inputSizez = inputSizez;
}

void Network::setInputLayer(Layer* layer)
{
	inputLayer = layer;
}

void Network::setOutputLayer(Layer* layer)
{
	outputLayer = layer;
}

void Network::addPerceptronLayer(int neuronsCount)
{
	hiddenLayers.push_back(new PerceptronLayer(neuronsCount));
}

void Network::addActivationLayer(string functionName)
{
	hiddenLayers.push_back(new ActivationLayer(functionName));
}

void Network::addConvolutionalLayer(int filtersCount)
{
	hiddenLayers.push_back(new ConvolutionalLayer(filtersCount));
}

void Network::build() {

	inputLayer->setupAsInputLayer(inputSizex, inputSizey, inputSizez);

	hiddenLayers[0]->setParentLayer(*inputLayer);

	for (int i = 1;i<hiddenLayers.size();i++) {
		hiddenLayers[i]->setParentLayer(*(hiddenLayers[i - 1]));
	}

	outputLayer->setParentLayer(*(hiddenLayers[hiddenLayers.size() - 1]));
}

struct FloatArray Network::forward(struct FloatArray &input) {
	
	// Bind inputs
	*(inputLayer->input) = input;
	inputLayer->forward();

	for (int i = 0; i < hiddenLayers.size(); i++) {
		hiddenLayers[i]->forward();
	}
	
	outputLayer->forward();

	return *(outputLayer->output);
}

void Network::backpropagate(struct FloatArray *error, float learningRate) {
	//ERROR
	FloatArray *error_ = outputLayer->backpropagate(error, learningRate);
	delete error;
	error = 0;
	error = error_;

	for (int i = hiddenLayers.size()-1; i >= 0; i--){
		error_ = hiddenLayers[i]->backpropagate(error, learningRate);
		delete error;
		error = 0;
		error = error_;
	}

	error_ = inputLayer->backpropagate(error, learningRate);
	delete error;
	error = 0;
	delete error_;
	error_ = 0;
}

void Network::save(string path) {
	ofstream ofile;
	ofile.open(path, ios::out);

	if (ofile.is_open()) {
		inputLayer->save(ofile);
		ofile << endl;
		for (int i = 0; i < hiddenLayers.size(); i++) {
			hiddenLayers[i]->save(ofile);		
			ofile << endl;
		}
		outputLayer->save(ofile);
		
	}

	ofile.close();
}

Network Network::retrieve(string path) {
	ifstream ifile;
	ifile.open(path, ios::in);

	int neuronIndex = -1;

	if (ifile.is_open()) {
		//cout << ifile << endl;
	}
	
	Network ntw(0, 0, 0);
	return ntw;
}

Network::~Network() {
	delete inputLayer->input;
	inputLayer->input = 0;
	delete inputLayer;
	inputLayer = 0;
	delete outputLayer;
	outputLayer = 0;

	for (int i = 0; i<hiddenLayers.size(); i++){
		delete hiddenLayers[i];
		hiddenLayers[i] = 0;
	}
}


