/*
 * The DataHandler class takes the arff file as input and returns the relevant data of the file for
 * the models to use such as:
 * 
 * size: the number of instances the arff file has. classIndex: the index of the class attribute.
 * 
 * classInstanceCount: number of class attributes the arff file has. split: all instances of the
 * 
 * dataset divided into a training set and a testing set.
 * 
 * trainingSplits: training split of the dataset.
 * 
 * testingSplits: testing split of the dataset.
 * 
 * classInstances: the list of all class attributes in a readable String format.
 */

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

      /* Set the class index as the index of the last attribute */
      classIndex = data.numAttributes() - 1;
      data.setClassIndex(classIndex);

      /* Set the size of the dataset as the number of instances*/
      size = data.numClasses();
      /* 
       * Set the number of attributes the class has given the number
       * of instances the dataset has
       */
      classInstanceCount = data.numInstances();
   }

   /*
    * Split the dataset into training and testing splits using the 10-fold cross validation method.
    */
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

   /*
    * Convert the data class value into a readable String format.
    */
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
