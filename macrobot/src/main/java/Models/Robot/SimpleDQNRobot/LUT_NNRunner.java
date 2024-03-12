package Models.Robot.SimpleDQNRobot;

import Models.LUT.StateActionTable;
import Models.NeuralNet.NN_OneHiddenLayer;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;


public class LUT_NNRunner {

    // Fixed value for learning rate and momentum
    public static final double LEARNING_RATE = 0.4;
    public static final double MOMENTUM_TERM = 0.8;

    public static void main(String[] args) throws IOException {

        // Initialize Neural Net structure
        NN_OneHiddenLayer nn = new NN_OneHiddenLayer(
                4,             // Input neuron number
                10,                     // Hidden neuron number
                1,                      // Output neuron number
                LEARNING_RATE,          // Learning rate (Rho)
                MOMENTUM_TERM,          // Momentum term (Alpha)
                -1,                     // Lower bound of weights
                1,                      // Upper bound of weights
                true                    // True/False: Bipolar/Binary
        );

        // Load LUT and normalize the Q-values
        StateActionTable trainedLUT = new StateActionTable(5, 5, 5, 5, 5);
        trainedLUT.load("out/statistics/LUT/LUTRobot_StateActionTable.txt");
        trainedLUT.normalizeQ();

        // Initialize parameters and NN weights
        int epoch = 0;
        double totalLoss;
        nn.initializeWeights();

        // Use LUT to train the weights of NN
        do {
            // Reset the total loss for each epoch
            totalLoss = 0;
            for (int a = 0; a < 5; a++) {
                for (int b = 0; b < 5; b++) {
                    for (int c = 0; c < 5; c++) {
                        for (int d = 0; d < 5; d++) {
                            for (int e = 0; e < 5; e++) {
                                // Normalize input states
                                double[] X = normalizeStates(a, b, c, d);
                                // Get the original index for each state
                                double[] lutIndex = {a, b, c, d, e};
                                // Train and calculate the total loss
                                double singleLoss = nn.train(X, trainedLUT.outputFor(lutIndex));
                                totalLoss += singleLoss;
                            }
                        }
                    }
                }
            }
            // Get the Root Mean Square Error of each epoch
            double totalError = Math.pow(totalLoss/3125, 0.5);
            System.out.println("Epoch: " + epoch + ", total error: " + totalError);
            // Epoch finished
            epoch++;
        } while (epoch < 1000);
        // Save the weights of NN
        File weights = new File("preTrainedWeights.txt");
        nn.saveWeights(weights);
        // End pre-training
        System.out.println("-------------------- NN pre-training is done! --------------------");
    }


    // Normalize the inputs of LUT for NN training
    private static double[] normalizeStates(int energy1, int dist1, int energy2, int dist2) {

        Map<Integer, Double> normalizeBipolar = new HashMap<Integer, Double>() {{
            put(0, -1.0);
            put(1, -0.5);
            put(2, 0.0);
            put(3, 0.5);
            put(4, 1.0);
        }};

        return new double[] {
                normalizeBipolar.get(energy1),
                normalizeBipolar.get(dist1),
                normalizeBipolar.get(energy2),
                normalizeBipolar.get(dist2),
                1.0                      // bias for NN training
        };
    }
}