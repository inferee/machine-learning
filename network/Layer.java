package network;

public class Layer {
	public final int nodes;
	public final int previous;
	public double[][] bias;
	public double[][] weight;

	public Layer(int previous, int nodes) {
		this.nodes = nodes;
		this.previous = previous;
		bias = Matrix.initGuassian(1, nodes, 0, 1);
		weight = Matrix.initGuassian(previous, nodes, 0, 1);
	}
	/*
	 * Takes a [1]x[previous nodes] matrix with input values as input and returns a
	 * [1]x[nodes] matrix with output values as output.
	 */

	public double[][] feed(double[][] input) {
		return Matrix.add(Matrix.multiply(input, weight), bias);
	}

	public double[][] inputDeriv(double[][] outputDeriv, double[][] thisInput) {
		return Matrix.multiplyElements(Matrix.sigmoidP(thisInput), outputDeriv);
	}

	public double[][] biasShift(double[][] inputDeriv, double learningRate) {
		return Matrix.multiply(inputDeriv, -learningRate);
	}

	public void biasChange(double[][] change) {
		Matrix.addTo(bias, change);
	}

	public double[][] weightShift(double[][] inputDeriv, double[][] prevOutput, double learningRate) {
		return Matrix.multiply(Matrix.multiply(Matrix.flip(prevOutput), inputDeriv), -learningRate);
	}

	public void weightChange(double[][] change) {
		Matrix.addTo(weight, change);
	}

	public double[][] prevOutDeriv(double[][] inputDeriv) {
		return Matrix.flip(Matrix.multiply(weight, Matrix.flip(inputDeriv)));
	}

	public double[][] train(double[][] outputDeriv, double[][] thisInput, double[][] prevOutput, double learningRate) {// TODO:
		double[][] inputDeriv = inputDeriv(outputDeriv, thisInput);
		Matrix.addTo(bias, biasShift(inputDeriv, learningRate));
		Matrix.addTo(weight, weightShift(inputDeriv, prevOutput, learningRate));
		return prevOutDeriv(inputDeriv);
	}
}
