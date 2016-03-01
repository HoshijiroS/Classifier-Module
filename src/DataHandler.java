import java.io.BufferedReader;
import java.io.IOException;

import weka.core.Instances;

public class DataHandler {
   private int size, classIndex, classInstanceCount;
   private Instances data;
   private Instances[][] split;
   private Instances[] trainingSplits;
   private Instances[] testingSplits;
   private String[] classInstances;

   public DataHandler(BufferedReader datafile) throws IOException {
      data = new Instances(datafile);

      classIndex = data.numAttributes() - 1;
      data.setClassIndex(classIndex);

      size = data.numClasses();
      classInstanceCount = data.numInstances();
   }

   public void crossValidationSplit(int numberOfFolds) {
      split = new Instances[2][numberOfFolds];

      for (int i = 0; i < numberOfFolds; i++) {
         split[0][i] = data.trainCV(numberOfFolds, i);
         split[1][i] = data.testCV(numberOfFolds, i);
      }

      trainingSplits = split[0];
      testingSplits = split[1];
   }

   public Instances[] getTrainingSplit() {
      return this.trainingSplits;
   }

   public Instances[] getTestingSplit() {
      return this.testingSplits;
   }

   public String[] getDataClasses() {
      classInstances = new String[size];

      for (int i = 0; i < size; i++) {
         classInstances[i] = data.classAttribute().value(i);
      }

      return classInstances;
   }

   public int getSize() {
      return this.size;
   }

   public int getClassIndex() {
      return this.classIndex;
   }

   public int getClassInstances() {
      return this.classInstanceCount;
   }

   public Instances getData() {
      return this.data;
   }
}
