package Models.XORLearning;

import Models.NeuralNet.NN_OneHiddenLayer;

import java.io.File;
import java.util.ArrayList;


public class XOR_NNRunner {

    NN_OneHiddenLayer nn;                        // NeuralNet for Binary or Bipolar representation
    static private double [][] trainInputVectors;       // Input vectors for XOR NN training
    static private double [] trainTargetVector;         // Target vectors for XOR NN training


    /** This method provides the TOTAL ERROR for each epoch.
     *  1. Input the NN you want to train
     *  2. Train the NN and sum the errors for each training pattern
     *  3. Output the TOTAL ERROR
     */
    private double getTotalError(NN_OneHiddenLayer nn) {
        // Initial the sum of errors
        double sumLoss = 0;

        // Train the NeuralNet and sum the error for each input pattern
        for (int i = 0; i < trainInputVectors.length; i++) {
            double singleLoss = nn.train(trainInputVectors[i], trainTargetVector[i]);
            // Sum the error of each training pattern
            sumLoss = sumLoss + singleLoss;
        }
        return sumLoss;
    }

    /** XOR problem using BINARY or BIPOLAR representation
     *  1. Initialize all WEIGHTS
     *  2. Start training
     *  3. Do not stop until TOTAL ERROR is less than 0.05 (print the TOTAL ERROR for each epoch)
     */
    public XOR_NNRunner(double setLearningRate, double setMomentum, boolean isBipolar, int totalRun) {

        /** Check if the type of representation (BINARY or BIPOLAR representation):
         *  Create NeuralNet (Input/Hidden/Output: 2/4/1) for training
         */
        if (!isBipolar) {
            // BINARY representation
            this.nn = new NN_OneHiddenLayer(2, 4, 1, setLearningRate, setMomentum, 0, 1, isBipolar);
            // Input vectors for BINARY representation (last column is for bias)
            trainInputVectors = new double[][] {
                    {0.0, 0.0, 1.0},
                    {0.0, 1.0, 1.0},
                    {1.0, 0.0, 1.0},
                    {1.0, 1.0, 1.0}
            };
            // Target vectors for BINARY representation
            trainTargetVector = new double[] {
                    0.0,
                    1.0,
                    1.0,
                    0.0
            };
        } else {
            // BIPOLAR representation
            this.nn = new NN_OneHiddenLayer(2, 4, 1, setLearningRate, setMomentum, -1, 1, isBipolar);
            // Input vectors for BIPOLAR representation (last column is for bias)
            trainInputVectors = new double[][] {
                    {-1.0, -1.0, 1.0},
                    {-1.0,  1.0, 1.0},
                    { 1.0, -1.0, 1.0},
                    { 1.0,  1.0, 1.0}
            };
            // Target vectors for BIPOLAR representation
            trainTargetVector = new double[]{
                    -1.0,
                    1.0,
                    1.0,
                    -1.0
            };
        }

        // Initialize all weights
        this.nn.initializeWeights();

        // Initialize totalError (The total error for each epoch)
        double totalError = 1.0;

        // Train for each input pattern and print the TOTAL ERROR for each epoch.
        if (totalRun == 1) {                     // Run only 1 loop
            for (int epoch = 1; totalError > 0.05; epoch++) {
                // TRAIN the NN and get the TOTAL ERROR for each epoch
                totalError = Math.pow(this.getTotalError(this.nn)/trainInputVectors.length, 0.5);
                // Print the TOTAL ERROR for each epoch
                if (totalError < 0.05) {
                    System.out.println(epoch + " " + totalError + "\n");
                } else {
                    System.out.println(epoch + " " + totalError);
                }
                // Record the total error of each epoch
                this.nn.arr.add("Epoch: " + epoch + ", total error: " + totalError);
            }
        } else {                                // Run more than 1 loop
            // For calculating the average epoch for multi-runs
            int totalEpochs = 0;

            for (int run = 0; run < totalRun; run++) {
                // Initialize all weights & totalError for each run
                this.nn.initializeWeights();
                totalError = 1.0;
                // Start each run
                for (int epoch = 1; totalError > 0.05; epoch++) {
                    // TRAIN the NN and get the TOTAL ERROR for each epoch
                    totalError = this.getTotalError(this.nn);
                    // Print the TOTAL EPOCHS for each run
                    if (totalError < 0.05) {
                        System.out.println(run + 1 + " " + epoch);
                    }
                    // Record the total error of each epoch
                    this.nn.arr.add("Epoch: " + epoch + ", total error: " + totalError);
                    totalEpochs++;
                }
                // Separate each run in the txt file and record the average epoch for the total runs
                if (run != totalRun - 1) {
                    this.nn.arr.add("---------------Next Run---------------");
                } else {
                    this.nn.arr.add("---------------" + totalRun + " Runs End---------------");
                    this.nn.arr.add("Average Epochs: " + totalEpochs/totalRun);
                }
            }
        }
        // Record the total error of each epoch for each run
        nn.save(new File("./out/statistics/XOR/Errors_Run_" + totalRun + "_Bipolar_" + isBipolar + "_LR_" + setLearningRate + "_MT_" + setMomentum + ".txt"));
    }

    public static void main(String args[]) {
        // NN training for XOR problem (Binary Representation without Momentum Term)
        System.out.println("XOR Binary Representation without Momentum Term" + "\n");
        System.out.println("Implement 1 run");                  // Implement 1 run
        XOR_NNRunner XOR_Runner_Binary_1run = new XOR_NNRunner(0.2, 0, false, 1);
        System.out.println("Implement 100 runs");               // Implement 100 runs
        XOR_NNRunner XOR_Runner_Binary_100runs = new XOR_NNRunner(0.2, 0, false, 100);

        // Implement NN training for XOR problem (Bipolar Representation without Momentum Term)
        System.out.println("\n" + "XOR Bipolar Representation without Momentum Term" + "\n");
        System.out.println("Implement 1 run");                  // Implement 1 run
        XOR_NNRunner XOR_Runner_Bipolar_1run = new XOR_NNRunner(0.2, 0, true, 1);
        System.out.println("Implement 100 runs");               // Implement 100 runs
        XOR_NNRunner XOR_Runner_Bipolar_100run = new XOR_NNRunner(0.2, 0, true, 100);

        // Implement NN training for XOR problem (Bipolar Representation with Momentum Term)
        System.out.println("\n" + "XOR Bipolar Representation with Momentum Term" + "\n");
        System.out.println("Implement 1 run");                  // Implement 1 run
        XOR_NNRunner XOR_Runner_BipolarMtm_1run = new XOR_NNRunner(0.2, 0.9, true,1);
        System.out.println("Implement 100 runs");               // Implement 100 runs
        XOR_NNRunner XOR_Runner_BipolarMtm_100run = new XOR_NNRunner(0.2, 0.9, true,100);

    }
}
