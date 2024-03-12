package Models.Interface;


/** Interface for the Neural Net (NN) Class
 *  @date 2 October 2023
 *  @author Kevin Chu
 */


public interface NeuralNetInterface extends CommonInterface{

    final double bias = 1.0;        // The input for each neuron bias weight


    /** This method implements a general sigmoid with asymptotes bounded by (a,b)
     *  Returns a sigmoid output of the input X
     */
    public double customSigmoid(double x);


    /** Initialize weights of SELECTED CURRENT weight layer to RANDOM VALUES.
     *  SELECTED PREVIOUS weight layer be the SAME as SELECTED CURRENT weight layer.
     */
    public void initialWeightsLayer(int layerIndex, int NumPrevLayer, int NumNextLayer);


    /** Initialize weights of ALL CURRENT weight layers to RANDOM VALUES.
     *  ALL PREVIOUS weight layer should be the SAME as ALL CURRENT weight layer
     *  (NO WEIGHT CHANGE at the FIRST training pattern).
     */
    public void initializeWeights();


    // Initialize weights of the SELECTED CURRENT & PREVIOUS weight layer to 0
    public void zeroWeightsLayer(int layerIndex, int NumPrevLayer, int NumNextLayer);


    // Initialize weights of ALL CURRENT & PREVIOUS weight layers to 0
    public void zeroWeights();


    //public void zeroErrorSignals();
}

