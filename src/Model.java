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

   public FastVector classify(Classifier model, Instances data) throws Exception {
      FastVector predictions = new FastVector();
      Evaluation evaluation = new Evaluation(data);

      model.buildClassifier(data);
      evaluation.crossValidateModel(model, data, 10, new Random(1));

      predictions = evaluation.predictions();
      System.out.println(evaluation.toSummaryString("---------------------------------\n "
            + model.getClass().getSimpleName() + "\n---------------------------------", false));

      return predictions;
   }

   public void setPredictions(Instances data, FastVector predictions) {
      NominalPrediction np;
      double predicted;

      predictionList = new String[data.numInstances()];

      for (int i = 1; i < predictions.size(); i++) {
         np = (NominalPrediction) predictions.elementAt(i);
         predicted = np.predicted();
         predictionList[i] = data.classAttribute().value((int) predicted);
      }
   }

   public String[] getPredictions() {
      return this.predictionList;
   }

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
