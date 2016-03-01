import java.io.BufferedReader;
import java.io.OutputStream;
import java.io.PrintStream;
import java.util.HashMap;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.lazy.IBk;
import weka.core.FastVector;
import weka.core.Instances;

public class Driver {

   public static void main(String[] args) throws Exception {
      FileHandler handler = new FileHandler();
      BufferedReader datafile = handler.readFile();

      DataHandler dataHandler = new DataHandler(datafile);

      // Get data classes
      String[] dataClasses = dataHandler.getDataClasses();
      int numClasses = dataHandler.getSize();
      int numInstances = dataHandler.getClassInstances();

      // System.out.println(dataClasses.length);

      // Use a set of classifiers
      Classifier[] models = {new NaiveBayes(), // Naive Bayes
            new LibSVM(), // SVM
            new MultilayerPerceptron(), // Neural Network
            new IBk(), // K-Nearest Neighbor
            new BayesNet() // Maximum Entropy

      };

      libsvm.svm.svm_set_print_string_function(new libsvm.svm_print_interface() {
         @Override
         public void print(String s) {} // Disables svm output
      });

      System.setErr(new PrintStream(new OutputStream() {
         public void write(int b) {}
      }));

      // Object[] pred;

      HashMap<Integer, Model> predictionPerModel = new HashMap<Integer, Model>();
      Instances data = dataHandler.getData();

      // Run for each model
      for (int j = 0; j < models.length; j++) {
         System.out.println("*********************************");
         Model model = new Model();

         // Collect every group of predictions for current model in a FastVector
         FastVector predictions = new FastVector();

         // For each training-testing split pair, train and test the classifier
         predictions = model.classify(models[j], data);

         model.calculateAccuracy(predictions);
         model.setPredictions(data, predictions);

         predictionPerModel.put(j, model);
         // System.out.println(predictionPerModel.get(1).getPredictions()[1]);
         System.out.println("*********************************");
      }

      Aggregator aggr =
            new Aggregator(models, predictionPerModel, dataClasses, numInstances, numClasses);
      aggr.initModelList();

      System.out.println("*********************************");
      // Majority Voting
      aggr.majorityVoting();
      System.out.println("*********************************");

      System.out.println("*********************************");
      // Weighted Majority Voting
      aggr.weightedMajorityVoting();
      System.out.println("*********************************");

      System.out.println("*********************************");
      // Stacking with SVM
      aggr.stackingWithSVM(data);
      System.out.println("*********************************");
   }
}
