package Models.NeuralNet;

import Models.Interface.NeuralNetInterface;
import com.fasterxml.jackson.databind.ObjectMapper;
import robocode.RobocodeFileWriter;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

/** This class only supports 1 HIDDEN LAYER.
 *  The TOTAL NUMBER OF LAYERS should always be 3.
 *  (Temporarily only supports 1 neuron for the OUTPUT LAYER,
 *  need to change the "outputFor" & "train" method to have multiple neurons for the OUTPUT LAYER)
 */
public class NN_OneHiddenLayer implements NeuralNetInterface{

    private int argNumInputs;                           // The number of inputs in your input vector
    private int argNumHidden;                           // The number of hidden neurons in your hidden
    // layer (Only 1 single hidden layer is supported)
    private int argNumOutputs;                          // The number of outputs in your output vector
    // (Temporarily should be 1)
    private double argLearningRate;                     // The learning rate coefficient
    private double argMomentumTerm;                     // The momentum coefficient
    private double argA;                                // Integer lower bound of sigmoid used by the
    // output neuron only
    private double argB;                                // Integer upper bound of sigmoid used by the
    // output neuron only
    private boolean argIsBipolar;                       // Determine BINARY or BIPOLAR representation,
    // FALSE for BINARY, TRUE for BIPOLAR

    private ArrayList<double [][]> currWeightLayers;    // ArrayList of 2D-arrays for CURRENT weights of
    // each layer
    private ArrayList<double [][]> prevWeightLayers;    // ArrayList of 2D-arrays for PREVIOUS weights of
    // each layer
    private ArrayList<double []> outputs;               // ArrayList of arrays for outputs of hidden &
    // output layer
    private ArrayList<double []> errSigs;               // ArrayList of arrays for errSigs of hidden &
    // output layer

    public static final double fixedWeightMin = -0.5;   // Fixed minimum for all weights
    public static final double fixedWeightMax = 0.5;    // Fixed maximum ot total layers
    private static final int fixedNumLayers = 3;        // Fixed Number ot total layers = 3

    // For record
    public ArrayList<String> logRecord = new ArrayList<>();
    private double[][] currWeights;

    public NN_OneHiddenLayer(int setNumIn, int setNumHidden, int setNumOut, double setLearningRate, double setMomentum, double setLB, double setUB, boolean isBipolar) {
        this.argNumInputs = setNumIn;
        this.argNumHidden = setNumHidden;
        this.argNumOutputs = setNumOut;
        this.argLearningRate = setLearningRate;
        this.argMomentumTerm = setMomentum;
        this.argA = setLB;
        this.argB = setUB;
        this.argIsBipolar = isBipolar;

        /** Create an ArrayList of 2D-arrays for saving the CURRENT WEIGHTS
         *  This class only supports 1 hidden layer (TOTAL number of layers must be 3,
         *  which means there are 2 layers of weights)
         */
        this.currWeightLayers = new ArrayList<>(fixedNumLayers - 1);
        // INPUT LAYER to HIDDEN LAYER (+1 for bias weight)
        this.currWeightLayers.add(0, new double [this.argNumHidden][this.argNumInputs + 1]);
        // HIDDEN LAYER to OUTPUT (+1 for bias weight)
        this.currWeightLayers.add(1, new double [this.argNumOutputs][this.argNumHidden + 1]);

        // Create an ArrayList of 2D-arrays for saving the PREVIOUS WEIGHTS
        this.prevWeightLayers = new ArrayList<>(fixedNumLayers - 1);
        // INPUT LAYER to HIDDEN LAYER (+1 for bias weight)
        this.prevWeightLayers.add(0, new double [this.argNumHidden][this.argNumInputs + 1]);
        // HIDDEN LAYER to OUTPUT (+1 for bias weight)
        this.prevWeightLayers.add(1, new double [this.argNumOutputs][this.argNumHidden + 1]);

        // Create an ArrayList of arrays for saving the OUTPUTS
        this.outputs = new ArrayList<>(fixedNumLayers - 1);
        this.outputs.add(0, new double [this.argNumHidden + 1]);        // HIDDEN LAYER (+1 for the bias)
        this.outputs.add(1, new double [this.argNumOutputs]);           // OUTPUT LAYER
        // Set Bias output value for the HIDDEN LAYER
        this.outputs.get(0)[this.argNumHidden] = bias;

        // Create an ArrayList of arrays for saving the ERROR SIGNALS
        this.errSigs = new ArrayList<>(fixedNumLayers - 1);
        this.errSigs.add(0, new double[this.argNumOutputs]);            // OUTPUT LAYER
        this.errSigs.add(1, new double[this.argNumHidden]);             // HIDDEN LAYER
    }


    /** This method implements a general sigmoid with asymptotes bounded by (a,b)
     *  Returns a sigmoid output of the input X
     */
    @Override
    public double customSigmoid(double x) {
        return (this.argB - this.argA) / (1 + Math.exp(-x)) + this.argA;
    }


    /** Initialize weights of SELECTED CURRENT weight layer to RANDOM VALUES.
     *  SELECTED PREVIOUS weight layer be the SAME as SELECTED CURRENT weight layer.
     */
    @Override
    public void initialWeightsLayer(int layerIndex, int NumPrevLayer, int NumNextLayer) {
        // Create a Random object
        Random random = new Random();
        // Assigns a random number (within a range) to each weight
        for (int i = 0; i < NumNextLayer; i++) {
            for (int j = 0; j < NumPrevLayer + 1; j++) {
                // This is for Java 17: this.currWeightLayers.get(layerIndex)[i][j] = random.nextDouble(fixedWeightMin, fixedWeightMax);
                this.currWeightLayers.get(layerIndex)[i][j] = fixedWeightMin + (fixedWeightMax - fixedWeightMin) * random.nextDouble();
                // Make the PREVIOUS weight layer be the SAME as the CURRENT weight layer
                this.prevWeightLayers.get(layerIndex)[i][j] = this.currWeightLayers.get(layerIndex)[i][j];
            }
        }
    }


    /** Initialize weights of ALL CURRENT weight layers to RANDOM VALUES.
     *  ALL PREVIOUS weight layer should be the SAME as ALL CURRENT weight layer
     *  (NO WEIGHT CHANGE at the FIRST training pattern).
     */
    @Override
    public void initializeWeights() {
        // Initialize all CURRENT WEIGHTS to random values,
        // and all PREVIOUS WEIGHTS to be the same as CURRENT WEIGHTS
        initialWeightsLayer(0, this.argNumInputs, this.argNumHidden);        // Input-to-Hidden layer
        initialWeightsLayer(1, this.argNumHidden, this.argNumOutputs);       // Hidden-to-Output layer
    }


    // Initialize weights of the SELECTED CURRENT & PREVIOUS weight layer to 0
    @Override
    public void zeroWeightsLayer(int layerIndex, int NumPrevLayer, int NumNextLayer) {
        // Assign 0 to each weight
        for (int i = 0; i < NumNextLayer; i++) {
            for (int j = 0; j < NumPrevLayer + 1; j++) {
                this.currWeightLayers.get(layerIndex)[i][j] = 0;
                this.prevWeightLayers.get(layerIndex)[i][j] = 0;
            }
        }
    }


    // Initialize weights of ALL CURRENT & PREVIOUS weight layers to 0
    @Override
    public void zeroWeights() {
        // Initialize all PREVIOUS & CURRENT WEIGHTS to 0
        zeroWeightsLayer(0, this.argNumInputs, this.argNumHidden);        // Input-to-Hidden layer
        zeroWeightsLayer(1, this.argNumHidden, this.argNumOutputs);       // Hidden-to-Output layer
    }


    /** This method is for FORWARD propagation step
     *  X is the input vector (an array of doubles)
     *  Returns the output value by the LUT or NN for this input vector
     *  This method only supports 1 output neuron
     */
    @Override
    public double outputFor(double [] X) {

        // Initialize the sum of weights
        double weightSum = 0;

        /** Forward propagation for the Input-to-Hidden layer:
         *  1. Sums the product of weights for each neuron
         *  2. Apply the sum to Sigmoid function to generate OUTPUT VALUE for each neuron
         *  3. Put the OUTPUT VALUES for each neuron into an array
         */
        for (int i = 0; i < this.argNumHidden; i++) {
            for (int j = 0; j < this.argNumInputs + 1; j++) {
                weightSum = weightSum + this.currWeightLayers.get(0)[i][j] * X[j];
            }
            this.outputs.get(0)[i] = this.customSigmoid(weightSum);
            // Must zero the weightSum before calculating the next neuron's output
            weightSum = 0;
        }
        // Forward propagation for the Hidden-to-Output layer
        for (int i = 0; i < this.argNumOutputs; i++) {
            for (int j = 0; j < this.argNumHidden + 1; j++) {
                weightSum = weightSum + this.currWeightLayers.get(1)[i][j] * this.outputs.get(0)[j];
            }
            this.outputs.get(1)[i] = this.customSigmoid(weightSum);
            // Must zero the weightSum before calculating the next neuron's output
            weightSum = 0;
        }
        return this.outputs.get(1)[0];
    }


    /** This method is for the whole TRAINING PROCESS, including FORWARD and BACKWARD propagation steps.
     *  Tell the LUT or NN the output value that should be mapped to the given input vector,
     *  i.e. the desired correct output value for an input.
     *  Give the input vector X & new target value argValue,
     *  the method returns the error in the output for that input vector.
     *  Steps:
     *  1. Implement FORWARD propagation, get ACTUAL OUTPUT for OUTPUT LAYER
     *  2. Calculate TOTAL ERROR
     *  3. Calculate ERROR SIGNAL(S) for OUTPUT LAYER
     *  4. Update WEIGHTS for HIDDEN-TO-OUTPUT LAYER (Save PREVIOUS WEIGHTS)
     *  5. Calculate ERROR SIGNAL(S) for HIDDEN LAYER
     *  6. Update WEIGHTS for INPUT-TO-HIDDEN LAYER
     *  7. Return TOTAL ERROR for this input set, and ready for the next input set training
     */
    @Override
    public double train(double [] X, double argValue) {

        // Implement FORWARD propagation,
        // get the ACTUAL OUTPUT value by the LUT or NN for this input vector
        double actualOutput = this.outputFor(X);

        // Calculate the TOTAL ERROR (LOSS) for each pattern in the training set (part of an epoch)
        double loss = Math.pow(actualOutput - argValue, 2);

        // Calculate ERROR SIGNAL(S) for OUTPUT LAYER
        if (!this.argIsBipolar) {                                   // BINARY representation
            for (int i = 0; i < this.argNumOutputs; i++) {
                this.errSigs.get(0)[i] = actualOutput * (1 - actualOutput) * (argValue - actualOutput);
            }
        } else {                                                    // BIPOLAR representation
            for (int i = 0; i < this.argNumOutputs; i++) {
                this.errSigs.get(0)[i] = 0.5 * (1 + actualOutput) * (1 - actualOutput) * (argValue - actualOutput);
            }
        }

        // Update the WEIGHTS of HIDDEN-TO-OUTPUT LAYER
        for (int i = 0; i < this.argNumOutputs; i++) {
            for (int j = 0; j < this.argNumHidden + 1; j++) {
                double weight = this.currWeightLayers.get(1)[i][j];
                weight = weight + this.argMomentumTerm * (this.currWeightLayers.get(1)[i][j] - this.prevWeightLayers.get(1)[i][j]) + this.argLearningRate * this.errSigs.get(0)[i] * this.outputs.get(0)[j];
                // Update PREVIOUS & CURRENT WEIGHTS
                this.prevWeightLayers.get(1)[i][j] = this.currWeightLayers.get(1)[i][j];
                this.currWeightLayers.get(1)[i][j] = weight;
            }
        }

        // Calculate ERROR SIGNAL(S) for HIDDEN LAYER
        for (int i = 0; i < this.argNumHidden; i++) {
            // Sum the PRODUCT OF CONNECTED ERROR SIGNALS & WEIGHTS (on output layer)
            double errorWeightSum = 0;
            for (int j = 0; j < this.argNumOutputs; j++) {
                errorWeightSum = errorWeightSum + this.errSigs.get(0)[j] * this.currWeightLayers.get(1)[j][i];
            }
            // Calculate and save the error signal(s)
            if (!this.argIsBipolar) {                               // BINARY representation
                this.errSigs.get(1)[i] = this.outputs.get(0)[i] * (1 - this.outputs.get(0)[i]) * errorWeightSum;
            } else {                                                // BIPOLAR representation
                this.errSigs.get(1)[i] = 0.5 * (1 + this.outputs.get(0)[i]) * (1 - this.outputs.get(0)[i]) * errorWeightSum;
            }
        }

        // Update the WEIGHTS of INPUT-TO-HIDDEN LAYER
        // (the WEIGHT CHANGE is always 0 for the first training pattern)
        for (int i = 0; i < this.argNumHidden; i++) {
            for (int j = 0; j < this.argNumInputs + 1; j++) {
                double weight = this.currWeightLayers.get(0)[i][j];
                weight = weight + this.argMomentumTerm * (this.currWeightLayers.get(0)[i][j] - this.prevWeightLayers.get(0)[i][j]) + this.argLearningRate * this.errSigs.get(1)[i] * X[j];
                // Update PREVIOUS & CURRENT WEIGHTS
                this.prevWeightLayers.get(0)[i][j] = this.currWeightLayers.get(0)[i][j];
                this.currWeightLayers.get(0)[i][j] = weight;
            }
        }

        return loss;
    }


    // Write the LUT or weights of NN to a file
    @Override
    public void save(File argFile) {

        // Initialize new file
        PrintStream file = null;

        //
        try{
            file = new PrintStream(new FileOutputStream(argFile) );
            for (int i = 0; i < logRecord.size(); i++) {
                file.println(logRecord.get(i));
            }
            file.flush();
            file.close();
        }
        catch(IOException e){
            System.out.println("There is an error, please check your inputs again.");
        }
    }


    /** Load the LUT or weights of NN from file.
     *  The load must have knowledge of how the data was written out by the save method.
     *  LUT or NN structure will be checked whether matching the data in the file.
     */
    @Override
    public void load(String argFileName) throws IOException {
        // Deserialize JSON file into Java object
        File file = new File(argFileName);
        ObjectMapper objMapper = new ObjectMapper();
        NN_OneHiddenLayer nnLoader = objMapper.readValue(file, NN_OneHiddenLayer.class);
        // Check whether the NN settings are identical
        if (nnLoader.argNumInputs != this.argNumInputs || nnLoader.argNumHidden != this.argNumHidden || nnLoader.argNumOutputs != this.argNumOutputs) {
            throw new IOException("NN structure does not match, please check the numbers of input/hidden/output neurons!");
        }
        if (nnLoader.argLearningRate != this.argLearningRate || nnLoader.argMomentumTerm != this.argMomentumTerm) {
            throw new IOException("Hyper-parameters do not match, please check the learning rate and momentum term!");
        }
        if (nnLoader.argIsBipolar != this.argIsBipolar) {
            throw new IOException("Representation does not match, please check the representation form again");
        }
        // Apply the weights of the loaded LUT or NN for further NN training
        this.currWeightLayers = nnLoader.currWeightLayers;
    }


    // Save NN weights
    public void saveWeights(File file) {
        // Initialize the parameters
        String[] weights = new String[2];
        int i = 0;
        int currLayerNum;
        // Save the NN weight for each layer
        for (currLayerNum = 0; currLayerNum != 2; currLayerNum++) {
            String weight = "";
            for (int j = 0; j < this.argNumHidden; j++) {
                for (int k = 0; k < this.argNumInputs + 1; k++) {
                    weight += currWeights[k][j] + "-";
                }
            }
            for (int j = 0; j < this.argNumOutputs; j++) {
                for (int k = 0; k < this.argNumHidden + 1; k++) {
                    weight += currWeights[k][j] + "-";
                }
            }
            weights[i] = weight;
            i++;
        }
        // Get the weights and save it to file
        try{
            RobocodeFileWriter weightsWriter = new RobocodeFileWriter(file.getAbsolutePath(), false);
            weightsWriter.write(weights[0] + "\r\n");
            weightsWriter.write(weights[1] + "\r\n");
            weightsWriter.close();
        }
        catch(Exception e){
            System.out.println(e);
        }
    }

    // Load NN weights
    public void loadWeights(File file) {
        // Initialize the parameters
        String[] weights = new String[2];
        // Load the NN weight for each layer
        try {
            BufferedReader reader = new BufferedReader(new FileReader(file.getAbsoluteFile()));
            for(int i = 0; i < 2; i++) {
                try {
                    weights[i] = reader.readLine();
                    System.out.println(weights[i]);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
            reader.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

        int i = 0;
        int currLayerNum;

        for(currLayerNum = 0; currLayerNum != 2; currLayerNum++) {
            String[] weight = Arrays.stream(weights[i].split("-")).filter(e -> e.trim().length() > 0).toArray(String[]::new);
            int id = 0;
            // Get the current weights for each layer
            if (currLayerNum == 0) {
                for (int j = 0; j < this.argNumHidden; j++) {
                    for (int k = 0; k < this.argNumInputs + 1; k++) {
                        currWeights[k][j] = Double.valueOf(weight[id]);
                        // Show the weights
                        System.out.print(currWeights[k][j] + "-");
                        id++;
                    }
                }
            } else if (currLayerNum == 1) {
                for (int j = 0; j < this.argNumOutputs; j++) {
                    for (int k = 0; k < this.argNumHidden + 1; k++) {
                        currWeights[k][j] = Double.valueOf(weight[id]);
                        // Show the weights
                        System.out.print(currWeights[k][j] + "-");
                        id++;
                    }
                }
            }
            System.out.println();
        }
        i++;
    }

}
