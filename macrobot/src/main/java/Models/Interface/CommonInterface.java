package Models.Interface;

import java.io.File;
import java.io.IOException;


/** This interface is common to both the Neural Net and LUT interfaces.
 *  The idea is that we should be able to easily switch the LUT
 *  for the Neural Net since the interfaces are identical.
 *  @date 2 October 2023
 *  @author Kevin Chu
 */


public interface CommonInterface {

    // Returns the output value by the LUT or NN for this input vector
    public double outputFor(double [] X);

    // Tell the LUT or NN the output value that should be mapped to the given input vector,
    public double train(double [] X, double argValue);


    // A method to write either a LUT or weights of an neural net to a file.
    public void save(File argFile) throws IOException;

    /** Loads the LUT or neural net weights from file. The load must of course
     *  have knowledge of how the data was written out by the save method.
     *  You should raise an error in the case that an attempt is being
     *  made to load data into an LUT or neural net whose structure does not match
     *  the data in the file. (e.g. wrong number of hidden neurons).
     */
    public void load(String argFileName) throws IOException;

}
