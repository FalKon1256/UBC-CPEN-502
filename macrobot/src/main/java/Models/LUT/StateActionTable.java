package Models.LUT;

import Models.Interface.LUTInterface;
import robocode.RobocodeFileOutputStream;
import java.io.*;


public class StateActionTable implements LUTInterface {

    private double [][][][][] lut;      // State & Action Look Up Table: 5 Dimensions, records all Q-values
    private int[][][][][] visits;       // Records the total visits of each State & Action
    private int numDim1Levels;          // 1st dimension
    private int numDim2Levels;          // 2nd dimension
    private int numDim3Levels;          // 3rd dimension
    private int numDim4Levels;          // 4th dimension
    private int numDim5Levels;          // 5th dimension

    private final int QVALUE_LB = -1;    // Lower bound for Q-value normalization (for NN training)
    private final int QVALUE_UB = 1;    // Upper bound for Q-value normalization (for NN training)


    public StateActionTable(
            int numDim1Levels,
            int numDim2Levels,
            int numDim3Levels,
            int numDim4Levels,
            int numDim5Levels) {

        this.numDim1Levels = numDim1Levels;
        this.numDim2Levels = numDim2Levels;
        this.numDim3Levels = numDim3Levels;
        this.numDim4Levels = numDim4Levels;
        this.numDim5Levels = numDim5Levels;

        lut = new double[numDim1Levels][numDim2Levels][numDim3Levels][numDim4Levels][numDim5Levels];
        visits = new int[numDim1Levels][numDim2Levels][numDim3Levels][numDim4Levels][numDim5Levels];
        this.initializeLUT();           // Initializes when creating the LUT
    }


    /** Initialize all Q-Values of the LUT to random values (equal to 0 but less than 1).
     *  Initialize the visit records (set to 0).
     */
    @Override
    public void initializeLUT() {
        for (int a = 0; a < numDim1Levels; a++) {
            for (int b = 0; b < numDim2Levels; b++) {
                for (int c = 0; c < numDim3Levels; c++) {
                    for (int d = 0; d < numDim4Levels; d++) {
                        for (int e = 0; e < numDim5Levels; e++) {
                            lut[a][b][c][d][e] = Math.random();
                            visits[a][b][c][d][e] = 0;
                        }
                    }
                }
            }
        }
    }


    // Returns the Q-value of the input State & Action
    @Override
    public double outputFor(double[] x) throws ArrayIndexOutOfBoundsException {
        if (x.length != 5) {
            throw new ArrayIndexOutOfBoundsException();
        } else {
            int a = (int)x[0];
            int b = (int)x[1];
            int c = (int)x[2];
            int d = (int)x[3];
            int e = (int)x[4];
            return lut[a][b][c][d][e];
        }
    }


    /** Updates the Q-value of the input State & Action to a new Q-value through training.
     *  Updates the visit record of the previous State & Action (input).
     */
    @Override
    public double train(double[] x, double target) throws ArrayIndexOutOfBoundsException {
        if (x.length != 5) {
            throw new ArrayIndexOutOfBoundsException();
        } else {
            int a = (int)x[0];
            int b = (int)x[1];
            int c = (int)x[2];
            int d = (int)x[3];
            int e = (int)x[4];
            lut[a][b][c][d][e] = target;
            visits[a][b][c][d][e]++;
        }
        return 1;
    }


    // This version saves the LUT in a format useful for training a NN
    @Override
    public void save(File filename) {
        System.out.println("*** Start Printing LUT...");
        PrintStream saveFile = null;

        try {
            saveFile = new PrintStream(new RobocodeFileOutputStream(filename));
        } catch (IOException e) {
            System.out.println("*** Could not create output stream for NN save file");
        }

        // First line is the number of rows of data
        saveFile.println(numDim1Levels * numDim2Levels * numDim3Levels * numDim4Levels * numDim5Levels);

        // Second line is the number of dimension per row
        saveFile.println(5);

        for (int a = 0; a < numDim1Levels; a++) {
            for (int b = 0; b < numDim2Levels; b++) {
                for (int c = 0; c < numDim3Levels; c++) {
                    for (int d = 0; d < numDim4Levels; d++) {
                        for (int e = 0; e < numDim5Levels; e++) {
                            // e, d, e2, d2, a, q visits
                            String row = String.format("%d, %d, %d, %d, %d, %2.3f, %d",
                                    a, b, c, d, e,
                                    lut[a][b][c][d][e],
                                    visits[a][b][c][d][e]
                            );
                            saveFile.println(row);
                        }
                    }
                }
            }
        }
        saveFile.close();
        System.out.println("*** Done Printing! Please use this format of LUT for NN training.");
    }


    // Loads the input LUT table to replace the current table.
    @Override
    public void load(String fileName) throws IOException {
        // Reads bytes from the file
        FileInputStream inputFile = new FileInputStream(fileName);
        // Reads characters from FileInputStream (bridge between byte and character streams)
        BufferedReader inputReader = new BufferedReader(new InputStreamReader(inputFile));
        int numTotalRows = numDim1Levels * numDim2Levels * numDim3Levels * numDim4Levels * numDim5Levels;

        // Reads the first line, and checks whether the number of rows is compatible.
        int numRows = Integer.valueOf(inputReader.readLine());
        // Reads the second line, and checks whether the number of dimensions is compatible.
        int numDimensions = Integer.valueOf(inputReader.readLine());

        if (numRows != numTotalRows || numDimensions != 5) {
            System.out.printf(
                    "*** rows/dimensions expected is %s/%s but %s/%s encountered\n",
                    numTotalRows, 5, numRows, numDimensions
            );
            inputReader.close();
            throw new IOException();
        }

        for (int a = 0; a < numDim1Levels; a++) {
            for (int b = 0; b < numDim2Levels; b++) {
                for (int c = 0; c < numDim3Levels; c++) {
                    for (int d = 0; d < numDim4Levels; d++) {
                        for (int e = 0; e < numDim5Levels; e++) {
                            // Reads each line with this format: e, d, e2, d2, a, q, visits
                            String line = inputReader.readLine();       // Reads each line at a time
                            String tokens[] = line.split(", ");   // Splits each value for each line
                            int dim1 = Integer.parseInt(tokens[0]);     // The State of each line
                            int dim2 = Integer.parseInt(tokens[1]);
                            int dim3 = Integer.parseInt(tokens[2]);
                            int dim4 = Integer.parseInt(tokens[3]);
                            int dim5 = Integer.parseInt(tokens[4]);     // The Action of each line
                            double q = Double.parseDouble(tokens[5]);   // The Q-value of each line
                            int v = Integer.parseInt(tokens[6]);        // The visit record of each line

                            lut[a][b][c][d][e] = q;
                            visits[a][b][c][d][e] = v;
                        }
                    }
                }
            }
        }
        inputReader.close();
    }


    // Normalize the Q-value of LUT for NN training
    public void normalizeQ() {
        for (int a = 0; a < numDim1Levels; a++) {
            for (int b = 0; b < numDim2Levels; b++) {
                for (int c = 0; c < numDim3Levels; c++) {
                    for (int d = 0; d < numDim4Levels; d++) {
                        for (int e = 0; e < numDim5Levels; e++) {

                            // Set all Q-value of LUT to the range of -1 to 1 (Bipolar)
                            lut[a][b][c][d][e] /= 20;
                            lut[a][b][c][d][e] = Math.max(lut[a][b][c][d][e], QVALUE_LB);
                            lut[a][b][c][d][e] = Math.min(lut[a][b][c][d][e], QVALUE_UB);


                            /** Another way to normalize
                            // Set Q-value minimum to -5
                            lut[a][b][c][d][e] = Math.max(lut[a][b][c][d][e], -5);
                            // Set Q-value maximum to 5
                            lut[a][b][c][d][e] = Math.min(lut[a][b][c][d][e], 5);
                            // Set all Q-value of LUT to the range of -1 to 1 (Bipolar)
                            lut[a][b][c][d][e] /= 5;
                             */

                            // Print the normalized LUT
                            System.out.println(lut[a][b][c][d][e]);
                        }
                    }
                }
            }
        }
    }

}



