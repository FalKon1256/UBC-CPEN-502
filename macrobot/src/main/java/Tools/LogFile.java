package Tools;

import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import robocode.RobocodeFileOutputStream;


/** This class implements a file logging mechanism.
 *  It is meant to enable diagnostic data to be written from a robocode tank to a file.
 */
public class LogFile {

    /** Private members of this class
     */
    public PrintStream stream;

    public LogFile(File argFile) {
        try {
            stream = new PrintStream(new RobocodeFileOutputStream(argFile));
            System.out.println("--+ Log file created.");
        } catch (IOException e) {
            System.out.println("*** IO exception during file creation attempt.");
        }
    }

    public void print(String argString) {
        stream.print(argString);
    }

    public void println(String argString) {
        stream.println(argString);
    }

}
