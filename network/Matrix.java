package network;

import java.util.Random;

public class Matrix {
	public static Random random = new Random();

	public static double[][] multiply(double[][] a, double[][] b) {
		if (!valid(a) || !valid(b) || a[0].length != b.length) {
			return null;
		}
		int r = a.length;
		int c = b[0].length;
		int num = a[0].length;
		double[][] result = new double[a.length][b[0].length];
		for (int i = 0; i < r; i++) {
			for (int j = 0; j < c; j++) {
				double sum = 0;
				for (int x = 0; x < num; x++) {
					sum += a[i][x] * b[x][j];
				}
				result[i][j] = sum;
			}
		}
		return result;
	}

	public static double[][] multiplyElements(double[][] a, double[][] b) {
		if (!valid(a) || !valid(b) || a.length != b.length || a[0].length != b[0].length) {
			return null;
		}
		double[][] result = new double[a.length][a[0].length];
		for (int i = 0; i < a.length; i++) {
			for (int j = 0; j < a[i].length; j++) {
				result[i][j] = a[i][j] * b[i][j];
			}
		}
		return result;
	}

	public static double[][] flip(double[][] a) {// modifies
		if (!valid(a)) {
			return null;
		}
		double[][] result = new double[a[0].length][a.length];
		for (int i = 0; i < a.length; i++) {
			for (int j = 0; j < a[i].length; j++) {
				result[j][i] = a[i][j];
			}
		}
		return result;
	}

	public static double[][] sigmoid(double[][] a) {
		if (!valid(a)) {
			return null;
		}
		double[][] result = new double[a.length][a[0].length];
		for (int i = 0; i < a.length; i++) {
			for (int j = 0; j < a[i].length; j++) {
				result[i][j] = sigmoid(a[i][j]);
			}
		}
		return result;
	}

	public static double[][] sigmoidP(double[][] a) {
		if (!valid(a)) {
			return null;
		}
		double[][] result = new double[a.length][a[0].length];
		for (int i = 0; i < a.length; i++) {
			for (int j = 0; j < a[i].length; j++) {
				result[i][j] = sigmoidP(a[i][j]);
			}
		}
		return result;
	}

	public static double[][] multiply(double[][] a, double val) {
		if (!valid(a)) {
			return null;
		}
		double[][] result = new double[a.length][a[0].length];
		for (int i = 0; i < a.length; i++) {
			for (int j = 0; j < a[i].length; j++) {
				result[i][j] = a[i][j] * val;
			}
		}
		return result;
	}

	public static double[][] multiplyTo(double[][] a, double val) {
		if (!valid(a)) {
			return null;
		}
		for (int i = 0; i < a.length; i++) {
			for (int j = 0; j < a[i].length; j++) {
				a[i][j] = a[i][j] * val;
			}
		}
		return a;
	}

	public static double[][] add(double[][] a, double[][] b) {
		if (!valid(a) || !valid(b) || a.length != b.length || a[0].length != b[0].length) {
			return null;
		}
		double[][] result = new double[a.length][a[0].length];
		for (int i = 0; i < a.length; i++) {
			for (int j = 0; j < a[i].length; j++) {
				result[i][j] = a[i][j] + b[i][j];
			}
		}
		return result;
	}

	public static double[][] addTo(double[][] a, double[][] b) {
		if (!valid(a) || !valid(b) || a.length != b.length || a[0].length != b[0].length) {
			return null;
		}
		for (int i = 0; i < a.length; i++) {
			for (int j = 0; j < a[i].length; j++) {
				a[i][j] += b[i][j];
			}
		}
		return a;
	}

	public static double[][] fill(int r, int c, double val) {
		double[][] result = new double[r][c];
		for (int i = 0; i < r; i++) {
			for (int j = 0; j < c; j++) {
				result[i][j] = val;
			}
		}
		return result;
	}

	public static double[][] initGuassian(int r, int c, double mean, double stDev) {
		double[][] result = new double[r][c];
		for (int i = 0; i < r; i++) {
			for (int j = 0; j < c; j++) {
				result[i][j] = random.nextGaussian() * stDev + mean;
			}
		}
		return result;
	}

	public static double[][] initUniform(int r, int c, double mean, double range) {
		double[][] result = new double[r][c];
		for (int i = 0; i < r; i++) {
			for (int j = 0; j < c; j++) {
				result[i][j] = random.nextDouble() * range + mean - range / 2;
			}
		}
		return result;
	}

	public static boolean valid(double[][] a) {
		return a != null && a.length > 0 && a[0].length > 0;
	}

	public static double sigmoid(double x) {
		return 1 / (Math.exp(-x) + 1);
	}

	public static double sigmoidP(double x) {
		double sig = sigmoid(x);
		return sig * (1 - sig);
	}

	public static void print(double[][] a) {
		if (a == null) {
			System.out.println("null");
			return;
		}
		for (int i = 0; i < a.length; i++) {
			for (int j = 0; j < a[i].length; j++) {
				System.out.print(a[i][j] + " ");
			}
			System.out.println();
		}
	}
}
