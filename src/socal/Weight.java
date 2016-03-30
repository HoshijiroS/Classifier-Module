package socal;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.HashMap;
import java.util.Iterator;

import org.apache.poi.ss.usermodel.Cell;
import org.apache.poi.ss.usermodel.Row;
import org.apache.poi.xssf.usermodel.XSSFSheet;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;

public final class Weight implements StringUtils {
   private final char TAG;
   private HashMap<String, Integer> weights;

   public Weight(String path, char tag) {
      this.TAG = tag;
      setWeights(path);
   }

   private void setWeights(String path) {
      weights = new HashMap<>();

      // Get the workbook instance for XLS file
      try (FileInputStream file = new FileInputStream(new File(path)); // Get first sheet from the workbook
           XSSFWorkbook workbook = new XSSFWorkbook(file)) {

         XSSFSheet sheet = workbook.getSheetAt(0);

         Iterator<Row> rowIterator = sheet.iterator();
         while (rowIterator.hasNext()) {
            Row row = rowIterator.next();

            Iterator<Cell> cellIterator = row.cellIterator();

            try {
               String word = format(cellIterator.next().getStringCellValue());
               int value = (int) cellIterator.next().getNumericCellValue();
               weights.put(word, value);
            } catch (IllegalStateException e) {
               System.err.println(e);
               rowIterator.next();
            }
         }
      } catch (IOException e) {
         System.err.println(e);
      }
   }

   @Override
   public String format(String word) {
      return word.toLowerCase().trim();
   }

   public int getWordValue(String key) {
      return weights.getOrDefault(key, 0);
   }

   public char getTag() {
      return TAG;
   }
}
