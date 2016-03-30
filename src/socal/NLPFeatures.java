package socal;

import java.io.IOException;
import java.util.Arrays;

public class NLPFeatures {

   public static void main(String[] args) {
      String inputPath     = "src\\Input\\Articles.xlsx";
      String outputPath    = "src\\Output\\";
      String stopwordsPath = "src\\Input\\Stopwords.txt";
      int outputs          = 20;

      PreprocessSO_CAL p = new PreprocessSO_CAL(inputPath, outputPath, stopwordsPath);
      
      try {
         p.excelOutput(outputs);
      } catch (IOException ex) {
         System.err.println(ex.getMessage());
      }
   }
}
