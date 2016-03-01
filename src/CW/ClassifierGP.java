/*Author: Chris Brown
* Date: 13/12/2015
* Description: Classifier that uses genetic programming*/

package CW;

import org.epochx.epox.Node;
import org.epochx.epox.Variable;
import org.epochx.epox.math.*;
import org.epochx.gp.model.GPModel;
import org.epochx.gp.representation.GPCandidateProgram;
import org.epochx.life.GenerationAdapter;
import org.epochx.life.Life;
import org.epochx.op.selection.TournamentSelector;
import org.epochx.representation.CandidateProgram;
import org.epochx.stats.Stats;
import java.util.ArrayList;
import java.util.List;

import static org.epochx.stats.StatField.GEN_FITNESS_MIN;
import static org.epochx.stats.StatField.GEN_FITTEST_PROGRAM;
import static org.epochx.stats.StatField.GEN_NUMBER;

public class ClassifierGP extends Classifier implements Cloneable {
    private Variable out1;
    private Variable out0;
    private Variable in1;
    private Variable in0;

    // The boolean inputs/outputs that we will test solutions against.
    private double[][] inputs;
    private double[] outputs;
    private int numAtt;
    private int numIns;

    private GPCandidateProgram gptree;

    public ClassifierGP(InstanceSet trainingSet) {

        class GPModelTree extends GPModel{
            @Override
            public double getFitness(CandidateProgram p) {
                GPCandidateProgram program = (GPCandidateProgram) p;
                gptree = program;

                double score = 0;
                for (int i=0; i<inputs.length; i++) {
                    // Set the variables.
                    in0.setValue(inputs[i][0]);
                    in1.setValue(inputs[i][1]);
                    out0.setValue((double)0);
                    out1.setValue((double)1);

                    int result = (int) ((double)program.evaluate() + 0.5);
                    if (result == outputs[i]) {
                        score++;
                    }
                }

                double rawFitness = numIns - score;
                return rawFitness; // + 0.1 * program.getProgramLength();
            }

            @Override
            public Class<?> getReturnType() {
                return Double.class;
            }
        }

        numAtt = Attributes.getNumAttributes();
        numIns = trainingSet.numInstances();

        inputs = new double[numIns][numAtt];
        outputs = new double[numIns];

        Instance [] instances = trainingSet.getInstances();

        //Move values into inputs and outputs arrays
        for(int i=0; i< instances.length; i++){
            double input1 = instances[i].getRealAttribute(0);
            double input2 = instances[i].getRealAttribute(1);
            double output = instances[i].getClassValue();

            inputs[i][0] = input1;
            inputs[i][1] = input2;
            outputs[i] = output;
        }

        // Construct the variables into fields.
        out1 = new Variable("out1",Double.class);
        out0 = new Variable("out0",Double.class);
        in1 = new Variable("in1",Double.class);
        in0 = new Variable("in0",Double.class);
        List<Node> syntax = new ArrayList<Node>();

        // Functions.
        syntax.add(new GreaterThanFunction());
        syntax.add(new LessThanFunction());
        syntax.add(new MultiplyFunction());
        syntax.add(new SubtractFunction());
        syntax.add(new AddFunction());
        syntax.add(new AbsoluteFunction());
        syntax.add(new SquareFunction());
        syntax.add(new SquareRootFunction());
        syntax.add(new DivisionProtectedFunction());
        syntax.add(new ModuloProtectedFunction());

        // Terminals.
        syntax.add(out1);
        syntax.add(out0);
        syntax.add(in1);
        syntax.add(in0);

        GPModel gp = new GPModelTree();
        gp.setSyntax(syntax);

        // Set parameters.
        gp.setPopulationSize(30);
        gp.setNoGenerations(50);
        gp.setNoElites(20);
        gp.setCrossoverProbability(0.9);
        gp.setMutationProbability(0.1);
        gp.setReproductionProbability(0.1);

        // Set operators and components.
        gp.setProgramSelector(new TournamentSelector(gp, 40));


        Life.get().addGenerationListener(new GenerationAdapter(){
            public void onGenerationEnd() {
                Stats.get().print(GEN_NUMBER, GEN_FITNESS_MIN);
            }
        });

        //train with training data
        gp.run();
    }

    public int classifyInstance(Instance ins) {

        in0.setValue(ins.getRealAttribute(0));
        in1.setValue(ins.getRealAttribute(1));

        int result = (int) (((double) gptree.evaluate()) + 0.5);
        if (result == ins.getClassValue()) return result;
        else return -1;
    }

    public void printClassifier() {
        System.out.println("Final Classifier");
        Stats.get().print(GEN_NUMBER, GEN_FITNESS_MIN, GEN_FITTEST_PROGRAM);
        gptree.toString();
    }
}
