import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.Stacking;
import weka.core.FastVector;
import weka.core.Instances;
import weka.classifiers.functions.LibSVM;

public class Aggregator {
   private Classifier[] models;
   private int[] classCounters;
   private String[] dataClasses;
   private HashMap<Integer, Model> predictionPerModel;
   private ArrayList<Model> modelList;
   private int numInstances;
   private int numClasses;

   public Aggregator(Classifier[] model, HashMap<Integer, Model> predictionPerModel,
                     String[] dataClasses, int numInstances, int numClasses) {
      this.models = model;
      this.predictionPerModel = predictionPerModel;
      this.numInstances = numInstances;
      this.modelList = new ArrayList<Model>();
      this.numClasses = numClasses;
      this.classCounters = new int[numClasses];
      this.dataClasses = dataClasses;
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

   public void majorityVoting() {
      System.out.println("---------------------------------");
      System.out.println("Majority Voting");
      System.out.println("---------------------------------");
      
      double likelihood = 0.0;
      int l = 1;

      while (l < numInstances) {
         for (int i = 0; i < modelList.size(); i++) {
            String[] classIds = modelList.get(i).getPredictions();

            for (int k = 0; k < numClasses; k++) {
               if (classIds[l] == dataClasses[k]) {
                  classCounters[k]++;
               }
            }
         }

         System.out.print("Instance [" + l + "]:");

         for (int m = 0; m < numClasses; m++) {
            if (classCounters[m] != 0) {
               likelihood = ((double) classCounters[m] / (double) models.length) * 100.0;
               System.out.print(
                     " " + dataClasses[m] + ": " + String.format("%.4f%%", likelihood) + " ");
               classCounters[m] = 0;
            }
         }
         System.out.println(" ");
         l++;
      }
   }

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

   public void weightedMajorityVoting() {
      System.out.println("---------------------------------");
      System.out.println("Weighted Majority Voting");
      System.out.println("---------------------------------");
      
      setWeights();
      double likelihood = 0.0;
      int weightedTotal = 0;
      int l = 1;

      while (l < numInstances) {
         for (int i = 0; i < modelList.size(); i++) {
            String[] classIds = modelList.get(i).getPredictions();
            weightedTotal = weightedTotal + modelList.get(i).getWeight();

            for (int k = 0; k < numClasses; k++) {
               if (classIds[l] == dataClasses[k]) {
                  classCounters[k] = classCounters[k] + modelList.get(i).getWeight();
               }
            }
         }

         System.out.print("Instance [" + l + "]:");

         for (int m = 0; m < numClasses; m++) {
            if (classCounters[m] != 0) {
               likelihood = ((double) classCounters[m] / (double) weightedTotal) * 100.0;
               System.out.print(
                     " " + dataClasses[m] + ": " + String.format("%.4f%%", likelihood) + " ");
               classCounters[m] = 0;
            }
         }
         System.out.println(" ");

         weightedTotal = 0;
         l++;
      }
   }

   public void stackingWithSVM(Instances trainingSet) throws Exception {
      // Set stacking classifier to SVM
      Stacking stackSVM = new Stacking();
      LibSVM libsvm = new LibSVM();
      Model model = new Model();
      stackSVM.setClassifiers(models);

      stackSVM.setMetaClassifier(libsvm);
      Evaluation eval = new Evaluation(trainingSet);
      eval.crossValidateModel(stackSVM, trainingSet, 10, new Random(1));
      System.out.println(eval.toSummaryString(
            "---------------------------------\n Stacking with SVM\n---------------------------------",
            false));

      FastVector predictions = eval.predictions();
      model.setPredictions(trainingSet, predictions);
      String[] predList = model.getPredictions();

      for (int i = 1; i < predList.length; i++) {
         System.out.println("Instance [" + i + "]: " + predList[i]);
      }
   }
}
