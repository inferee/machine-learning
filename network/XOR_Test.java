package network;

public class XOR_Test {
	public Layer[] layers;
	public double learningRate;
	public double[][][] input = new double[][][] { { { 0, 0 } }, { { 0, 1 } }, { { 1, 0 } }, { { 1, 1 } } };
	public double[] goal = new double[] { 0, 1, 1, 0 };

	public XOR_Test(int cycles) {
		layers = new Layer[2];
		layers[0] = new Layer(3, 2);
		layers[1] = new Layer(1, 3);
		learningRate = 0.25;
		double[][][] layerOutputs = new double[3][][];
		double[][][] layerInputs = new double[2][][];
//		 System.out.println("Layer 0 Bias");
//		 Matrix.print(layers[0].bias);
//		 System.out.println("Layer 0 Weight");
//		 Matrix.print(layers[0].weight);
//		 System.out.println("Layer 1 Bias");
//		 Matrix.print(layers[1].bias);
//		 System.out.println("Layer 1 Weight");
//		 Matrix.print(layers[1].weight);
		while (cycles-- > 0) {
			for (int i = 0; i < 4; i++) {
				layerOutputs[0] = input[i];
				layerInputs[0] = layers[0].feed(layerOutputs[0]);
				layerOutputs[1] = Matrix.sigmoid(layerInputs[0]);
				layerInputs[1] = layers[1].feed(layerOutputs[1]);
				layerOutputs[2] = Matrix.sigmoid(layerInputs[1]);
				double[][] deriv = new double[][] { { 2 * layerOutputs[2][0][0] - 2 * goal[i] } };
				deriv = layers[1].train(deriv, layerInputs[1], layerOutputs[1], learningRate);
				deriv = layers[0].train(deriv, layerInputs[0], layerOutputs[0], learningRate);
				System.out.println(layerOutputs[2][0][0] + " " + goal[i]);
			}
		}
//		 System.out.println("Layer 0 Bias");
//		 Matrix.print(layers[0].bias);
//		 System.out.println("Layer 0 Weight");
//		 Matrix.print(layers[0].weight);
//		 System.out.println("Layer 1 Bias");
//		 Matrix.print(layers[1].bias);
//		 System.out.println("Layer 1 Weight");
//		 Matrix.print(layers[1].weight);
	}

	public static void main(String[] args) {
		new XOR_Test(10000);
	}
}
