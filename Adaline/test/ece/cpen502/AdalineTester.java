package ece.cpen502;

import org.junit.Assert;
import org.junit.Test;

public class AdalineTester {

    @Test
    public void testOutput() {

        double [] inputVector = {0.5, -0.6};
        double [] testWeights = {1.2, 0.04, -0.96};

        Adaline testAdaline = new Adaline(2);
        testAdaline.setWeights(testWeights);

        double expectedOutput = 1.796;
        double actualOutput = testAdaline.output(inputVector);

        Assert.assertEquals(expectedOutput, actualOutput, 0.001);
    }

    @Test
    public void testLoss() {
        double [][] trainInputVectors = {
                {0.0, 0.0},
                {0.0, 1.0},
                {1.0, 0.0},
                {1.0, 1.0}
        };

        double [] trainTargetVector = {
                0.0,
                0.0,
                0.0,
                1.0
        };

        double [] testWeights = {1.2, 0.04, -0.96};
        Adaline testAdaline = new Adaline(2);
        testAdaline.setWeights(testWeights);

        double expectedLoss = 3.5536;
        double actualLoss = testAdaline.loss(trainInputVectors, trainTargetVector);

        Assert.assertEquals(expectedLoss, actualLoss, 0.001);
    }

}
