#include "public.h"
#include <iostream>

using namespace std;

int main(int argc, char *argv[]){

	// Configure network
	Network network(2, 1, 1);
	
	network.setInputLayer(new PerceptronLayer(3));
	network.addActivationLayer("sigmoid");
	network.addPerceptronLayer(1);
	network.setOutputLayer(new ActivationLayer("sigmoid"));

	network.build();

	// XOR
	FloatArray input1(2, 1, 1);
	FloatArray output1(1, 1, 1);
	input1.array[0] = 0;
	input1.array[1]  = 1;
	output1.array[0] = 1;

	FloatArray input2(2, 1, 1);
	FloatArray output2(1, 1, 1);
	input2.array[0] = 1;
	input2.array[1] = 0;
	output2.array[0] = 1;

	FloatArray input3(2, 1, 1);
	FloatArray output3(1, 1, 1);
	input3.array[0] = 1;
	input3.array[1] = 1;
	output3.array[0] = 0;

	FloatArray input4(2, 1, 1);
	FloatArray output4(1, 1, 1);
	input4.array[0] = 0;
	input4.array[1] = 0;
	output4.array[0] = 0;

	FloatArray output1__(network.forward(input1));		
	FloatArray output2__(network.forward(input2));		
	FloatArray output3__(network.forward(input3));		
	FloatArray output4__(network.forward(input4));		
		
	cout << "INPUT 0 1 " << (output1__)(0, 0, 0) << endl;
	cout << "INPUT 1 0 " << (output2__)(0, 0, 0) << endl;
	cout << "INPUT 1 1 " << (output3__)(0, 0, 0) << endl;
	cout << "INPUT 0 0 " << (output4__)(0, 0, 0) << endl;
	cout << endl;
	FloatArray* gradient;


	network.save("start.txt");
	for (int i = 0;i<200000;i++) {

		FloatArray output1_(network.forward(input1));		
		gradient=quadraticErrorGradient(output1_, output1);
	//	cout << quadraticError(*output1_, output1) << " ";
		network.backpropagate(gradient, 1);

		FloatArray output2_(network.forward(input2));		
		gradient=quadraticErrorGradient(output2_, output1);
	//	cout << quadraticError(*output2_, output2) << " ";
		network.backpropagate(gradient, 1);

		FloatArray output3_(network.forward(input3));		
		gradient=quadraticErrorGradient(output3_, output3);
	//	cout << quadraticError(*output3_, output3) << " ";
		network.backpropagate(gradient, 1);

		FloatArray output4_(network.forward(input4));		
		gradient=quadraticErrorGradient(output4_, output4);
	//	cout << quadraticError(*output4_, output4) << endl;
		network.backpropagate(gradient, 1);

	}

	network.save("end.txt");
	FloatArray output1_(network.forward(input1));		
	FloatArray output2_(network.forward(input2));		
	FloatArray output3_(network.forward(input3));		
	FloatArray output4_(network.forward(input4));		
		

	cout << "INPUT 0 1 " << (output1_)(0, 0, 0) << endl;
	cout << "INPUT 1 0 " << (output2_)(0, 0, 0) << endl;
	cout << "INPUT 1 1 " << (output3_)(0, 0, 0) << endl;
	cout << "INPUT 0 0 " << (output4_)(0, 0, 0) << endl;

}
