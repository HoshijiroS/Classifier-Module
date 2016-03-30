/*
 * The Driver class handles the execution of the whole classifier module
 */

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Scanner;

import org.apache.poi.xssf.usermodel.XSSFRow;
import org.apache.poi.xssf.usermodel.XSSFSheet;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;

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
	   
      // Read the arff file and allow the DataHandler class to process
      // it for necessary parameters
      FileHandler handler = new FileHandler();
      BufferedReader datafile = handler.readFile();

      DataHandler dataHandler = new DataHandler(datafile);
      
      // Train data classes using 10-fold cross validation
      //Instances data = dataHandler.crossValidationSplit(10);
      Instances data = dataHandler.getData();
      
      // Get data classes
      String[] dataClasses = dataHandler.getDataClasses();
      
//      PrintStream out = new PrintStream(new FileOutputStream("output.txt"));
//      System.setOut(out);
      
      
      int numClasses = dataHandler.getSize();
      int numInstances = dataHandler.getClassInstances();

      // Load models here
      
      // Use a set of 5 classifiers
      Classifier[] models = new Classifier[4];{new NaiveBayes(), // Naive Bayes
            new LibSVM(), // SVM
//            new MultilayerPerceptron(), // Neural Network
            new IBk(), // K-Nearest Neighbor
            new BayesNet() // Maximum Entropy

      };
      
      libsvm.svm.svm_set_print_string_function(new libsvm.svm_print_interface() {
         @Override
         // Disables the geeky SVM output
         public void print(String s) {}
      });

      /*System.setErr(new PrintStream(new OutputStream() {
         // Disables the warnings returned by the classifiers
         public void write(int b) {}
      }));*/

      HashMap<Integer, Model> predictionPerModel = new HashMap<Integer, Model>();

      // Store every group of predictions for current model in a FastVector
      FastVector predictions = new FastVector();

      // Run for each model
      for (int j = 0; j < models.length; j++) {
         System.out.println("*********************************");
         Model model = new Model(models[j].getClass().getSimpleName());

         // For each training-testing split pair, train and test the classifier
         predictions = model.classify(data);
         models[j] = model.getModel();

         // Get and set the accuracy of the models given their predictions
         //model.calculateAccuracy(predictions);
         model.setPredictions(data, predictions);
         
         predictionPerModel.put(j, model);
         System.out.println("*********************************");
      }
      
      /*
       * Rule Based Classifier Models
       */

      Model termFrequencyModel = new Model("Term Frequency Model");
      termFrequencyModel.setWeight(0);
      Model socalModel = new Model("SOCAL");
      socalModel.setWeight(0);
      
      //Read carmen's 330 annotated articles
      File file = new File("Input/Carmen330Sample.xlsx");
      XSSFWorkbook wb;
      try {
         wb = new XSSFWorkbook(file);
         XSSFSheet sheet = wb.getSheetAt(0);
         
         ArrayList<String> articles = new ArrayList<String>();
         ArrayList<String> sentiments = new ArrayList<String>();
         Iterator rows = sheet.iterator();
         rows.next();
         while(rows.hasNext()){
           XSSFRow row = (XSSFRow) rows.next();
           if(row.getCell(0) != null){
             articles.add(row.getCell(0).getStringCellValue());
             sentiments.add(row.getCell(1).getStringCellValue());
           }
         }
         
         //Create new TermFrequency
         TermFrequency tf = new TermFrequency(articles, sentiments);
         
         //Set values for TermFrequency's Model
//         termFrequencyModel.setAccuracy(tf.getAccuracy());
//         termFrequencyModel.setPredictionList(tf.getClassifierPredictions());
         
         //Create new SOCAL
//         SOCAL sc = new SOCAL(articles, sentiments);
//         socalModel.setAccuracy(sc.getAccuracy());
//         socalModel.setPredictionList(sc.getClassifierPredictions());
         
      }catch(Exception e){
        e.printStackTrace();
      }
      
      int lastModel = models.length-1;
//      predictionPerModel.put(++lastModel, termFrequencyModel);
//      predictionPerModel.put(++lastModel, socalModel);     
      
      

      // Aggregate the predictions made by the set of classifiers
      Aggregator aggr =
            new Aggregator(models, predictionPerModel, dataClasses, numInstances, numClasses, predictions);
      aggr.populateModelList();

      // Stores the list of aggregated predictions
      double[] aggrPredictions;
      double aggrAccuracy = 0.0;
      
      System.out.println("*********************************");
      // Majority Voting
      aggrPredictions = aggr.majorityVoting();
      
      // Get accuracy
      aggrAccuracy = aggr.calculateAggrAccuracy(aggrPredictions);
      
      System.out.println("---------------------------------");
      System.out.println("Accuracy: " + String.format("%.4f%%", aggrAccuracy));
      System.out.println("*********************************");
       
      System.out.println("*********************************");
      // Weighted Majority Voting
      aggrPredictions = aggr.weightedMajorityVoting();
      
      // Get accuracy
      aggrAccuracy = aggr.calculateAggrAccuracy(aggrPredictions);
      
      System.out.println("---------------------------------");
      System.out.println("Accuracy: " + String.format("%.4f%%", aggrAccuracy));
      System.out.println("*********************************");
      
      System.out.println("*********************************");
      // Stacking with SVM
      aggr.stackingWithSVM(data);
      System.out.println("*********************************");
   }
}
