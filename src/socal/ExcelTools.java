package socal;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.Set;
import org.apache.poi.ss.usermodel.Row;
import org.apache.poi.ss.usermodel.Sheet;
import org.apache.poi.xssf.streaming.SXSSFWorkbook;

public class ExcelTools{
  
   public static void makeExcelOutput(String[][]data, Set<String> keys, String outputPath) 
      throws FileNotFoundException, IOException {
      
      //Keep 100 rows in memory, exceeding rows will be flushed to disk
      SXSSFWorkbook workbook = new SXSSFWorkbook(100);
      Sheet sheet            = workbook.createSheet();
      
      makeOutputHeader(sheet, keys);
      makeOutputRows(sheet, keys, data);
      
      closeCreatedFile(workbook, outputPath);
   }

   private static void makeOutputHeader(Sheet sheet, Set<String> keys) {
      Row header     = sheet.createRow(0);
      int hCellCount = 0;
      header.createCell(hCellCount++).setCellValue("Article");
      
      for (String ngram : keys) {
         header.createCell(hCellCount++).setCellValue(ngram);
      }
   }
   
   private static void makeOutputRows(Sheet sheet, Set<String> key, String[][] data) {
      for (int i = 0; i < data.length; i++) {
         String article = Arrays.toString(data[i]);
         
         Row row        = sheet.createRow(i + 1);
         int rCellCount = 0;
         row.createCell(rCellCount++).setCellValue(article);
         
         for (String keys : key) {
            row.createCell(rCellCount++).setCellValue(article.contains(keys) ? 1 : 0);
         }//Places 1 if the ngram is present in the article, 0 otherwise
      }
   }
      
   private static void closeCreatedFile(SXSSFWorkbook workbook, String outputPath) throws FileNotFoundException, IOException{
      try (FileOutputStream out = new FileOutputStream(outputPath)) {
         workbook.write(out);
      }
      workbook.dispose(); // dispose of temporary files backing this workbook on disk
   }
}
