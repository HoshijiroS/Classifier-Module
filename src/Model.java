/**
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

   /** Classify instances */
   public FastVector classify(Classifier model, Instances data) throws Exception {
      FastVector predictions = new FastVector();
      Evaluation evaluation = new Evaluation(data);

      model.buildClassifier(data);
      /** Use 10-fold cross validation to train the model */
      evaluation.crossValidateModel(model, data, 10, new Random(1));

      /** Output data regarding the model such as: kappa statistic, mean absolute error, etc */
      predictions = evaluation.predictions();
      System.out.println(evaluation.toSummaryString("---------------------------------\n "
            + model.getClass().getSimpleName() + "\n---------------------------------", false));

      return predictions;
   }

   /** Convert the predictions (returned as double values) into a readable String value */
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

   /**
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
}
