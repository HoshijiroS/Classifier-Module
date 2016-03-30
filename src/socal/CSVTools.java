package socal;

import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Set;

public class CSVTools {
   private static final char NEW_LINE = '\n';
   
   public static void makeCSVOutput(String[][] data, Set<String> keys, String outputPath)
           throws FileNotFoundException, IOException {
      FileWriter writer = new FileWriter(outputPath);

      makeOutputHeader(writer, keys);
      makeOutputRows(writer, keys, data);

      closeCreatedFile(writer);
   }

   private static void append(StringBuilder sb, String text) throws IOException{
      char COMMA    = ',';
      sb.append(text);
      sb.append(COMMA);
   }
   
   private static void makeOutputHeader(FileWriter writer, Set<String> keys) throws IOException {
      StringBuilder sb = new StringBuilder();
      append(sb, "Article");
      
      for (String key : keys) {
         append(sb, key);
      }
      sb.append(NEW_LINE);
      writer.append(sb);
   }

   private static void makeOutputRows(FileWriter writer, Set<String> keys, String[][] data) throws IOException {
      StringBuilder sb = new StringBuilder();
      for (int i = 0; i < data.length; i++) {
         String article = Arrays.toString(data[i]);
         append(sb, "Article "+i);

         for (String key : keys) {
            append(sb, article.contains(key) ? "1" : "0");
         }
         sb.append(NEW_LINE);
      }
      writer.append(sb);
   }

   private static void closeCreatedFile(FileWriter writer) throws IOException {
      writer.flush();
      writer.close();
   }
}
