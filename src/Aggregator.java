/*
 * The Aggregator class takes the input of other models in order to produce classifications with
 * varying probabilities and a single classification made by a meta-classifier.
 * 
 * Parameters are: model: the models to be aggregated.
 * 
 * predictionPerModel: predictions produced by each model.
 * 
 * dataClasses: classes (in String) of the arff file.
 * 
 * numInstances: the number of instances the arff file has.
 * 
 * numClasses: number of classes the arff file has.
 */

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.meta.Stacking;
import weka.core.FastVector;
import weka.core.Instances;
import weka.classifiers.functions.LibSVM;

public class Aggregator {
   private Classifier[] models;
   private int[] classCounters;
   private String[] dataClasses;
   private double[] predictions;
   private HashMap<Integer, Model> predictionPerModel;
   private ArrayList<Model> modelList;
   private int numInstances;
   private int numClasses;
   private double likelihood;
   private double temp;
   private double aggrPred;
   private int weightedTotal;
   private int instance;

   public Aggregator(Classifier[] model, HashMap<Integer, Model> predictionPerModel,
         String[] dataClasses, int numInstances, int numClasses) {
      this.models = model;
      this.predictionPerModel = predictionPerModel;
      this.numInstances = numInstances;
      this.modelList = new ArrayList<Model>();
      this.numClasses = numClasses;
      this.classCounters = new int[numClasses];
      this.dataClasses = dataClasses;
      this.predictions = new double[numInstances];
      this.likelihood = 0.0;
      this.temp = 0.0;
      this.aggrPred = 0.0;
      this.weightedTotal = 0;
      this.instance = 0;
   }

   public void initClassCounters() {
      for (int i = 0; i < classCounters.length; i++) {
         classCounters[i] = 0;
      }
   }

   public void initModelList() {
      for (int i = 0; i < this.models.length; i++) {
         modelList.add(predictionPerModel.get(i));
      }
   }

   /*
    * Get the accuracy of the aggregator by comparing the prediction it made against the actual
    * classification of the instance and computing how many times it makes the correct
    * classification over the number of predictions made.
    */
   public double calculateAggrAccuracy(FastVector predictions, double[] aggrPredictions) {
      double correct = 0;

      for (int i = 0; i < predictions.size(); i++) {
         NominalPrediction np = (NominalPrediction) predictions.elementAt(i);
         if (aggrPredictions[i] == np.actual()) {
            correct++;
         }
      }

      return 100 * correct / predictions.size();
   }

   /*
    * Majority Voting takes the predictions made by the models and counts the number of votes each
    * class received. Outputs are in probabilities in the case that the models have casted different
    * votes.
    */
   public double[] majorityVoting() {
      System.out.println("---------------------------------");
      System.out.println("Majority Voting");
      System.out.println("---------------------------------");

      // Tally predictions made by the models
      while (instance < numInstances) {
         for (int i = 0; i < modelList.size(); i++) {
            String[] classIds = modelList.get(i).getPredictions();

            for (int k = 0; k < numClasses; k++) {
               if (classIds[instance] == dataClasses[k]) {
                  classCounters[k]++;
               }
            }
         }

         // Display the probabilities per instance
         System.out.print("Instance [" + (instance + 1) + "]:");

         for (int m = 0; m < numClasses; m++) {
            if (classCounters[m] != 0) {
               likelihood = ((double) classCounters[m] / (double) models.length) * 100.0;
               System.out.print(
                     " " + dataClasses[m] + ": " + String.format("%.4f%%", likelihood) + " ");
               classCounters[m] = 0;

               // Get the aggregated prediction by taking the predicted
               // class with the highest likelihood value
               if (temp < likelihood) {
                  temp = likelihood;
                  aggrPred = (double) m;
               }

               classCounters[m] = 0;
            }
         }
         System.out.println(" ");

         // Add aggregated prediction to list
         predictions[instance] = aggrPred;
         System.out.println("Final Prediction: " + dataClasses[(int) predictions[instance]]);
         System.out.println(" ");

         // Initialize values
         temp = 0.0;

         instance++;
      }
      return predictions;
   }

   /*
    * This method is used in order to determine the weights the models will be assigned with during
    * the Majority Voting phase. Weights were determined depending on the model's produced accuracy.
    */
   public void setWeights() {
      int accuracy = 0;

      for (int i = 0; i < models.length; i++) {
         accuracy = (int) modelList.get(i).getAccuracy();

         if (accuracy > 90) {
            modelList.get(i).setWeight(9);
         }

         else if (accuracy > 80) {
            modelList.get(i).setWeight(8);
         }

         else if (accuracy > 70) {
            modelList.get(i).setWeight(7);
         }

         else if (accuracy > 60) {
            modelList.get(i).setWeight(6);
         }

         else if (accuracy > 50) {
            modelList.get(i).setWeight(5);
         }

         else if (accuracy > 40) {
            modelList.get(i).setWeight(4);
         }

         else if (accuracy > 30) {
            modelList.get(i).setWeight(3);
         }

         else if (accuracy > 20) {
            modelList.get(i).setWeight(2);
         }

         else if (accuracy > 10) {
            modelList.get(i).setWeight(1);
         }
      }
   }

   /*
    * Weighted Majority Voting takes the predictions made by the models and counts the number of
    * votes each class received. This time, weights are assigned to each vote a model casts. Outputs
    * are in probabilities in the case that the models have casted different votes.
    */
   public double[] weightedMajorityVoting() {
      System.out.println("---------------------------------");
      System.out.println("Weighted Majority Voting");
      System.out.println("---------------------------------");

      setWeights();

      instance = 0;

      // Tally the weighted votes the models casted
      while (instance < numInstances) {
         for (int i = 0; i < modelList.size(); i++) {
            String[] classIds = modelList.get(i).getPredictions();
            weightedTotal = weightedTotal + modelList.get(i).getWeight();

            for (int k = 0; k < numClasses; k++) {
               if (classIds[instance] == dataClasses[k]) {
                  classCounters[k] = classCounters[k] + modelList.get(i).getWeight();
               }
            }
         }

         // Display the probabilities per instance
         System.out.print("Instance [" + (instance + 1) + "]:");

         for (int m = 0; m < numClasses; m++) {
            if (classCounters[m] != 0) {
               likelihood = ((double) classCounters[m] / (double) weightedTotal) * 100.0;
               System.out.print(
                     " " + dataClasses[m] + ": " + String.format("%.4f%%", likelihood) + " ");

               // Get the aggregated prediction by taking the predicted
               // class with the highest likelihood value
               if (temp < likelihood) {
                  temp = likelihood;
                  aggrPred = (double) m;
               }

               classCounters[m] = 0;
            }
         }
         System.out.println(" ");

         // Add aggregated prediction to list
         predictions[instance] = aggrPred;
         System.out.println("Final Prediction: " + dataClasses[(int) predictions[instance]]);
         System.out.println(" ");

         // Initialize values
         temp = 0.0;
         weightedTotal = 0;

         instance++;
      }
      return predictions;
   }

   /*
    * Stacking with SVM takes the predictions made by the models and uses them as a feature set. The
    * meta-classifier used is the SVM, trained using 10-fold cross validation. Outputs are a single
    * class prediction made by the meta-classifier.
    */
   public void stackingWithSVM(Instances trainingSet) throws Exception {
      // Set stacking classifier to SVM
      Stacking stackSVM = new Stacking();
      LibSVM libsvm = new LibSVM();
      Model model = new Model();
      stackSVM.setClassifiers(models);

      stackSVM.setMetaClassifier(libsvm);
      Evaluation eval = new Evaluation(trainingSet);

      // Use 10-fold cross validation in order to train the meta-classifier
      eval.crossValidateModel(stackSVM, trainingSet, 10, new Random(1));
      System.out.println(eval.toSummaryString(
            "---------------------------------\n Stacking with SVM\n---------------------------------",
            false));

      // Get predictions made by the meta-classifier
      FastVector predictions = eval.predictions();
      model.setPredictions(trainingSet, predictions);
      String[] predList = model.getPredictions();

      // Display the prediction per instance
      for (int i = 0; i < predList.length; i++) {
         System.out.println("Instance [" + (i + 1) + "]: " + predList[i]);
         System.out.println(" ");
      }
   }
}
