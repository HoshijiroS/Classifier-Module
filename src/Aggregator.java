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
   private String[] modDataClasses;
   private double[] modPredictions;
   private double[] aggrPredictions;
   private HashMap<Integer, Model> predictionPerModel;
   private ArrayList<Model> modelList;
   private double[][] likelihoodPerInstance;
   private double[] likelihoodList;
   private FastVector predictions;
   private int numInstances;
   private int numClasses;
   private double likelihood;
   private double temp;
   private double aggrPred;
   private int tieCount;

   public Aggregator(Classifier[] model, HashMap<Integer, Model> predictionPerModel,
         String[] dataClasses, int numInstances, int numClasses, FastVector predictions) {
      this.models = model;
      this.predictionPerModel = predictionPerModel;
      this.numInstances = numInstances;
      this.predictions = predictions;
      this.modelList = new ArrayList<Model>();
      this.likelihoodPerInstance = new double[numInstances][numClasses];
      this.likelihoodList = new double[numInstances];
      this.numClasses = numClasses;
      this.classCounters = new int[numClasses];
      this.dataClasses = dataClasses;
      this.modDataClasses = new String[dataClasses.length + 1];
      this.modPredictions = new double[numInstances];
      this.aggrPredictions = new double[numInstances];
      this.likelihood = 0.0;
      this.temp = 0.0;
      this.aggrPred = 0.0;
      this.tieCount = 0;
   }

   public void initClassCounters() {
      for (int i = 0; i < classCounters.length; i++) {
         classCounters[i] = 0;
      }
   }

   public void populateModelList() {
      for (int i = 0; i < this.models.length; i++) {
         modelList.add(predictionPerModel.get(i));
      }
   }

   public void populateModifiedPredList() {
      for (int i = 0; i < dataClasses.length; i++) {
         modDataClasses[i] = dataClasses[i];
      }

      modDataClasses[dataClasses.length] = "NONE";
   }

   /*
    * Get the accuracy of the aggregator by comparing the prediction it made against the actual
    * classification of the instance and computing how many times it makes the correct
    * classification over the number of predictions made.
    */
   public double calculateAggrAccuracy(double[] aggrPredictions) {
      double correct = 0;
      int ties = 0;
      
      for (int i = 0; i < this.predictions.size(); i++) {
         NominalPrediction np = (NominalPrediction) this.predictions.elementAt(i);
         if(!(aggrPredictions[i] == dataClasses.length)){
            if (aggrPredictions[i] == np.actual()) {
               correct++;
            }
         }
         else {
            ties++;
         }
      }

      System.out.println("Ties found: " + ties);
      return 100 * correct / this.predictions.size();
   }

   public double[] classify(int config) {
      int weightTotal = 0;
      
      setWeights(config);
      // Tally predictions made by the models
      for (int instance = 0; instance < numInstances; instance++) {
         for (int i = 0; i < modelList.size(); i++) {
            String[] classIds = modelList.get(i).getPredictions();
            weightTotal = weightTotal + modelList.get(i).getWeight();

            for (int k = 0; k < numClasses; k++) {
               if (classIds[instance] == dataClasses[k]) {
                  classCounters[k] = classCounters[k] + modelList.get(i).getWeight();
               }
            }
         }

         // Display the probabilities per instance
         System.out.print("Instance [" + (instance + 1) + "]:");

         for (int i = 0; i < numClasses; i++) {
            likelihoodPerInstance[instance][i] = -1;
            if (classCounters[i] != 0) {
               likelihood = ((double) classCounters[i] / (double) weightTotal) * 100.0;
               likelihoodPerInstance[instance][i] = likelihood;
               System.out.print(
                     " " + dataClasses[i] + ": " + String.format("%.4f%%", likelihood) + " ");

               // Get the aggregated prediction by taking the predicted
               // class with the highest likelihood value
               if (temp < likelihood) {
                  temp = likelihood;
                  aggrPred = (double) i;
                  likelihoodList[instance] = temp;
               }

               classCounters[i] = 0;
            }
         }
         System.out.println(" ");

         // Add aggregated prediction to list
         modPredictions[instance] = aggrPred;
         aggrPredictions[instance] = aggrPred;
         
         // Check for ties
         populateModifiedPredList();
         for (int i = 0; i < dataClasses.length; i++) {
            if (likelihoodList[instance] == likelihoodPerInstance[instance][i]) {
               tieCount++;
               if (tieCount > 1) {
                  modPredictions[instance] = dataClasses.length;
               }
            }
         }

         System.out.println("Final Prediction: " + modDataClasses[(int) modPredictions[instance]]);
         System.out.println(" ");

         // Initialize values
         temp = 0.0;
         weightTotal = 0;
         tieCount = 0;
      }
      
      double aggrAccuracy = calculateAggrAccuracy(modPredictions);
      System.out.println("---------------------------------");
      System.out.println("Modified Accuracy: " + String.format("%.4f%%", aggrAccuracy));
      
      return aggrPredictions;
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

      return classify(0);
   }

   /*
    * This method is used in order to determine the weights the models will be assigned with during
    * the Majority Voting phase. Weights were determined depending on the model's produced accuracy.
    */
   public void setWeights(int config) {
      int accuracy = 0;

      for (int i = 0; i < models.length; i++) {
         if (config == 1) {
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
         } else if (config == 0) {
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

      return classify(1);
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
