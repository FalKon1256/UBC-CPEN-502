package ece.cpen502;

public class Adaline {

    private int numInputs;
    private double [] weights;

    public Adaline(int numInputs) {
        this.numInputs = numInputs;
        weights = new double [numInputs + 1]; // +1 is to accommodate a bias weight
    }

    public double output(double [] inputVector) {
        if (weights.length != inputVector.length + 1)
            throw new ArrayIndexOutOfBoundsException();
        else {
            double weightedSum = 0.0;
            weightedSum += weights[0];
            for (int i=1; i<this.weights.length; i++) {
                weightedSum += weights[i] * inputVector[i-1];
            }

            return weightedSum;
        }
    }

    public void setWeights(double [] weightVector) {
        if (weights.length != weightVector.length)
            throw new ArrayIndexOutOfBoundsException();
        else
            for (int i=0; i<weightVector.length; i++) {
                weights[i] = weightVector[i];
            }
    }

    public double loss(double [][] trainInputVectors, double [] trainTargetOutputs) {
        double loss = 0.0;
        for (int i=0; i<trainTargetOutputs.length; i++) {
            double y = this.output(trainInputVectors[i]);
            loss += Math.pow(y - trainTargetOutputs[i], 2);
        }
        return loss;
    }

}
