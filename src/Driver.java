/*
 * The Driver class handles the execution of the whole classifier module
  */

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
      /*
       * Read the arff file and allow the DataHandler class to process
       * it for necessary parameters
        */
      FileHandler handler = new FileHandler();
      BufferedReader datafile = handler.readFile();

      DataHandler dataHandler = new DataHandler(datafile);

      /* 
       * Get data classes 
       */
      String[] dataClasses = dataHandler.getDataClasses();

      int numClasses = dataHandler.getSize();
      int numInstances = dataHandler.getClassInstances();

      // Use a set of 5 classifiers
      Classifier[] models = {
	     new NaiveBayes(), // Naive Bayes
         // new LibSVM(), // SVM
         // new MultilayerPerceptron(), // Neural Network
         new IBk(), // K-Nearest Neighbor
         new BayesNet() // Maximum Entropy
      };

      libsvm.svm.svm_set_print_string_function(
		  new libsvm.svm_print_interface() {
         @Override
         /* 
          * Disables the geeky SVM output 
          */
         public void print(String s) {}
      });

      System.setErr(new PrintStream(new OutputStream() {
         /* 
          * Disables the warnings returned by the classifiers
          */
         public void write(int b) {}
      }));

      HashMap<Integer, Model> predictionPerModel = new HashMap<>();
      Instances data = dataHandler.getData();

      /* 
       * Store every group of predictions for current model in a FastVector 
       */
      FastVector predictions = new FastVector();

      /* 
       * Run for each model 
       */
      for (int j = 0; j < models.length; j++) {
         System.out.println("*********************************");
         Model model = new Model();

         /* 
          * For each training-testing split pair, train and test the classifier 
          */
         predictions = model.classify(models[j], data);

         /* 
          * Get and set the accuracy of the models given their predictions 
          */
         model.calculateAccuracy(predictions);
         model.setPredictions(data, predictions);

         predictionPerModel.put(j, model);
         System.out.println("*********************************");
      }

      /*
       * Aggregate the predictions made by the set of classifiers
       */
      Aggregator aggr =
        new Aggregator(models, predictionPerModel, dataClasses, 
    		numInstances, numClasses, predictions);
      aggr.populateModelList();

      /*
       * Stores the list of aggregated predictions 
       */
      double[] aggrPredictions;
      double aggrAccuracy = 0.0;

      System.out.println("*********************************");
      /* 
       * Majority Voting 
       */
      aggrPredictions = aggr.majorityVoting();

      /* 
       * Get accuracy 
       */
      aggrAccuracy = aggr.calculateAggrAccuracy(aggrPredictions);

      System.out.println("---------------------------------");
      System.out.println("Accuracy: " + String.format("%.4f%%", aggrAccuracy));
      System.out.println("*********************************");

      System.out.println("*********************************");
      /* 
       * Weighted Majority Voting 
       */
      aggrPredictions = aggr.weightedMajorityVoting();

      /* 
       * Get accuracy 
       */
      aggrAccuracy = aggr.calculateAggrAccuracy(aggrPredictions);

      System.out.println("---------------------------------");
      System.out.println("Accuracy: " + String.format("%.4f%%", aggrAccuracy));
      System.out.println("*********************************");

      System.out.println("*********************************");
      /* 
       * Stacking with SVM 
       */
      aggr.stackingWithSVM(data);
      System.out.println("*********************************");
   }
}
