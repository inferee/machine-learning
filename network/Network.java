package network;

public class Network {
	private Layer[] layers;
	private int[] nodes;
	private int numLayers;
	private double learningRate;

	public Network(int[] nodes, double learningRate) {
		this.nodes = nodes;
		this.learningRate = learningRate;
		numLayers = nodes.length - 1;
		layers = new Layer[numLayers];
		for (int i = 0; i < numLayers; i++) {
			layers[i] = new Layer(nodes[i + 1], nodes[i]);
		}
	}
	// [inputs][1][inputlength] inputs
	// [1][outputLength] goals

	public void train(double[][][] inputs, double[][][] goals) {// currently xor converges to total loss of 1
		if (inputs.length != goals.length) {
			return;
		}
		double[][][][] outputs = new double[inputs.length][numLayers + 1][][];
		double[][][][] inbetween = new double[inputs.length][numLayers][][];
		double[][] avDeriv = new double[1][goals[0][0].length];
		double loss = 0;
		for (int x = 0; x < inputs.length; x++) {
			outputs[x][0] = inputs[x];
			for (int i = 0; i < numLayers; i++) {
				inbetween[x][i] = layers[i].feed(outputs[x][i]);
				outputs[x][i + 1] = Matrix.sigmoid(inbetween[x][i]);
			}
			double[][] deriv = Matrix.multiplyTo(Matrix.add(outputs[x][numLayers], Matrix.multiply(goals[x], -1)), 2);
			for (int i = 0; i < nodes[numLayers]; i++) {
				loss += Math.pow(outputs[x][numLayers][0][i] - goals[x][0][i], 2);
			}
			Matrix.addTo(avDeriv, deriv);
		}
		System.out.println(loss);
		Matrix.multiplyTo(avDeriv, 1.0 / inputs.length);
		double[][][] avWeightChange = new double[numLayers][][];
		double[][][] avBiasChange = new double[numLayers][][];
		for (int x = inputs.length - 1; x >= 0; x--) {
			double[][] currDeriv = avDeriv;
			for (int i = numLayers - 1; i >= 0; i--) {
				currDeriv = layers[i].inputDeriv(currDeriv, inbetween[x][i]);
				Matrix.addTo(avBiasChange[i], layers[i].biasShift(currDeriv, learningRate));
				Matrix.addTo(avWeightChange[i], layers[i].weightShift(currDeriv, outputs[x][i], learningRate));
				currDeriv = layers[i].train(currDeriv, inbetween[x][i], outputs[x][i], learningRate);
			}
		}
		for (int i = 0; i < numLayers; i++) {
			Matrix.multiplyTo(avWeightChange[i], 1.0 / inputs.length);
			Matrix.multiplyTo(avBiasChange[i], 1.0 / inputs.length);
			layers[i].weightChange(avWeightChange[i]);
			layers[i].biasChange(avBiasChange[i]);
		}
	}

	public static void main(String[] args) {
		Network xor = new Network(new int[] { 2, 4, 1 }, 0.5);
		double[][][] inputs = new double[][][] { { { 0, 0 } }, { { 0, 1 } }, { { 1, 0 } }, { { 1, 1 } } };
		double[][][] outputs = new double[][][] { { { 0 } }, { { 1 } }, { { 1 } }, { { 0 } } };
		for (int i = 0; i < 1000; i++) {
			xor.train(inputs, outputs);
		}
	}
}
