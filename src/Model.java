/*
 * The Model class produces predictions of a certain instance's classification based on the features
 * it is given.
 * 
 * Each model has: predictionList: a list of predictions (in String) made by the model on each
 * instance.
 * 
 * accuracy: the accuracy of the model in predicting the class of a certain instance.
 * 
 * weight: a weight assigned to the model based on its accuracy.
 */

import java.io.File;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.NominalPrediction;
import weka.core.FastVector;
import weka.core.Instances;

public class Model {
   private String[] predictionList;
   private double accuracy;
   private int weight;
   private String name;
   Classifier model;
   
   public Model(String name) {
	   this.name = name;
	   Classifier model = null;
   }   
   
   // Classify instances
   public FastVector classify(Instances data) throws Exception {
      FastVector predictions = new FastVector();
      Evaluation evaluation = new Evaluation(data);
      
      if(new File(Paths.MODELS_DIR + this.name +".model").exists()){
    	  System.out.println("LOAD MODEL.");
    	  model = (Classifier) weka.core.SerializationHelper.read(Paths.MODELS_DIR + this.name + ".model");
      }
    	/* 
      else{
    	  new File(Paths.MODELS_DIR + this.name +".model").createNewFile();
    	  model.buildClassifier(data);
    	  weka.core.SerializationHelper.write(Paths.MODELS_DIR + this.name +".model", model);    	  
      }*/
      
      // Use 10-fold cross validation to train the model
      //evaluation.crossValidateModel(model, data, 10, new Random(1));
      
      evaluation.evaluateModel(model, data);
      
      // Output data regarding the model such as: kappa statistic,
      // mean absolute error, etc
      predictions = evaluation.predictions();
      System.out.println(evaluation.toSummaryString("---------------------------------\n "
            + model.getClass().getSimpleName() + "\n---------------------------------", false));

      return predictions;
   }
   
   public Classifier getModel(){
	   return model;
   }

   /*
    * Convert the predictions (returned as double values) into a readable String value
    */
   public void setPredictions(Instances data, FastVector predictions) {
      NominalPrediction np;
      double predicted;

      predictionList = new String[data.numInstances()];

      for (int i = 0; i < predictions.size(); i++) {
         np = (NominalPrediction) predictions.elementAt(i);
         predicted = np.predicted();
         predictionList[i] = data.classAttribute().value((int) predicted);
      }
   }

   public String[] getPredictions() {
      return this.predictionList;
   }

   /*
    * Get the accuracy of each model by comparing the prediction against the actual classification
    * of the instance and computing how many times it makes the correct classification over the
    * number of predictions made.
    */
   public void calculateAccuracy(FastVector predictions) {
      double correct = 0;

      for (int i = 0; i < predictions.size(); i++) {
         NominalPrediction np = (NominalPrediction) predictions.elementAt(i);
         if (np.predicted() == np.actual()) {
            correct++;
         }
      }

      accuracy = 100 * correct / predictions.size();
   }

   public double getAccuracy() {
      return this.accuracy;
   }

   public void setWeight(int weight) {
      this.weight = weight;
   }

   public int getWeight() {
      return this.weight;
   }
   
   public String getName(){
	   return this.name;
   }
   
   //For Rule Based
   public void setAccuracy(double accuracy){
     this.accuracy = accuracy;
   }
   
   public void setPredictionList(String[] predictionList){
     this.predictionList = predictionList;
   }
}
