/*Author: Chris Brown
* Date: 13/12/2015
* Description: Classifier that uses the framework Neuroph to implement a classification Neural Network*/

package CW;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;

import org.neuroph.core.Layer;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.util.TransferFunctionType;

public class ClassifierNN extends Classifier implements Cloneable {
    private int numAtt;
    private MultiLayerPerceptron nnet;

    private static String trainPerfInfo = "";


    public ClassifierNN(InstanceSet trainingSet) {

        numAtt = Attributes.getNumAttributes();

        Instance [] instances = trainingSet.getInstances();
        DataSet trainSet = new DataSet(numAtt,1);

        //Add each instance into DataSet
        for(int i=0; i< instances.length; i++){
            double input1 = instances[i].getRealAttribute(0);
            double input2 = instances[i].getRealAttribute(1);
            double output = instances[i].getClassValue();

            DataSetRow tempRow = new DataSetRow(new double[]{input1, input2}, new double[]{output});
            trainSet.addRow(tempRow);
        }

        // create multi layer perceptron
        nnet = new MultiLayerPerceptron(TransferFunctionType.TANH, 2, 4, 1);

        nnet.getLearningRule().addListener(new LearningListener());
        nnet.getLearningRule().setLearningRate(0.01);
        nnet.getLearningRule().setMaxIterations(40);

        //learn training set
        nnet.learn(trainSet);
        System.out.print("Trained the NN");
        //Store performance info
        try{
            FileWriter fw = new FileWriter("NNTrainingResults2.csv");
            fw.append(trainPerfInfo);
            fw.close();
        }
        catch(IOException e){
            System.out.println(e.getMessage());
        }
    }

    static class LearningListener implements LearningEventListener {


        long start = System.currentTimeMillis();

        public void handleLearningEvent(LearningEvent event) {
            BackPropagation bp = (BackPropagation) event.getSource();
            System.out.println("Current iteration: " + bp.getCurrentIteration());
            System.out.println("Error: " + bp.getTotalNetworkError());
            System.out.println((System.currentTimeMillis() - start) / 1000.0);

            //Update performance string
            trainPerfInfo = trainPerfInfo + bp.getCurrentIteration() + "," + bp.getTotalNetworkError() + "\n";

            start = System.currentTimeMillis();
        }

    }

    public int classifyInstance(Instance ins) {

        //make dataRow
        double[] ats = ins.getRealAttributes();
        DataSetRow dataRow = new DataSetRow(new double[]{ats[0], ats[1]}, new double[]{ins.getClassValue()});

        //Get output and check
        nnet.setInput(dataRow.getInput());
        nnet.calculate();
        int output = (int)(nnet.getOutput()[0] + 0.5);
        if(output == dataRow.getDesiredOutput()[0]) return output;
        else return -1;

    }

    public void printClassifier() {
        String nn = nnet.getLabel();

        System.out.println(" Number of Layers = " + nnet.getLayersCount() + " ->");
        Layer[] layers = nnet.getLayers();
        for(int i = 0; i < layers.length; i++){
            System.out.println("\tLayer " + i + " has " + layers[i].getNeuronsCount() + " Neuron(s)");
        }
        System.out.println("Output Error = " + nnet.getOutputNeurons()[0].getError());
    }
}