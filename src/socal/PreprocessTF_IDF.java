package socal;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

public class PreprocessTF_IDF extends Preprocess {
   private HashMap<String, Double> corpusWords;

   public PreprocessTF_IDF(String inputPath, String outputPath, String stopwordsPath) {
      super(inputPath, outputPath, stopwordsPath);
      
      setCorpusWords();
      removeStopWords(corpusWords.keySet());
      removeInvalidWords(corpusWords);
   }

//<editor-fold defaultstate="collapsed" desc="TF IDF Outputs">
   public void excelOutput(double outputs) throws IOException {
      output(outputs, true);
   }
   
   public void csvOutput(double outputs) throws IOException {
      output(outputs, false);
   }
   
   @Override
   public void output(double outputs, boolean isExcel) throws IOException {
      for (double i = 0; i < outputs; i++) {
         double percentage = Math.floor(1.0 * (i / outputs) * 100);
         Set<String> keys = removeLowPercentageWords(percentage);
         
         if (isExcel) {
            ExcelTools.makeExcelOutput(data, keys, outputPath + percentage + "%.xslx");
         } else {
            CSVTools.makeCSVOutput(data, keys, outputPath + percentage + "%.csv");
         }
      }
   }
//</editor-fold>
   
//<editor-fold defaultstate="collapsed" desc="Removers">
   private void removeInvalidWords(HashMap<String, Double> corpusWords) {
      //Remove words of length 1 except for "I"
      corpusWords.keySet().removeIf(s->s.length() <= 1 && !s.equalsIgnoreCase("i"));

      //Remove the "-" from words Ex: "-anyos" -> "anyos", "--frj" -> "frj"
      //Remove the words which begin in "-"
      HashMap<String, Double> temp = new HashMap<>();
      corpusWords.entrySet().stream()
              .filter(entry->entry.getKey().startsWith("-"))
              .forEach((entry)->{
                 temp.put(entry.getKey(), entry.getValue());
              });
      corpusWords.keySet().removeAll(temp.keySet());

      //Place the words without their "-"
      temp.entrySet().stream()
              .forEach((entry)->{
                 String key = entry.getKey();
                 while (key.startsWith("-")) {
                    key = key.substring(1);
                 }
                 corpusWords.put(key, entry.getValue());
              });
   }   

   private Set<String> removeLowPercentageWords(double percentage) {
      int cutoff = (int) Math.floor(corpusWords.size() * percentage / 100);
      double valCutoff;

      List<Double> values = corpusWords.entrySet().stream()
         .sorted(Map.Entry.comparingByValue()) //Sort by value
         .map(Map.Entry::getValue)             //Reference only the map's values
         .collect(Collectors.toList());        //Return only the values

      double temp = Double.MAX_VALUE;
      for (int i = 0; i < values.size(); i++) {
         if (i == cutoff) {
            temp = values.get(i);   //Change temp to be the value at cutoff
            break;
         }
      }                             //Because only final variables can
      valCutoff = temp;               //be used in Lambda

      //Create a tempMap so that the original corpus can be reused
      HashMap<String, Double> tempMap = new HashMap<>(corpusWords);
      tempMap.values().removeIf(v->v < valCutoff);  //Remove low percentage entries

      return tempMap.keySet();
   }
//</editor-fold>

   public HashMap<String, Double> getCorpusWords() {
      return corpusWords;
   }

   private void setCorpusWords() {
      corpusWords      = new HashMap<>();
      int articleCount = data.length;

      for (String key : getUniqueWords()) {
         double value  = 0;

         TfIdfCorpus calculator = new TfIdfCorpus(data, key);
         for (int j = 0; j < articleCount; j++) {
            value += calculator.tfIdf(getDataAtIndex(j), key);
         }
         corpusWords.put(key, value);
      }
   }
}
