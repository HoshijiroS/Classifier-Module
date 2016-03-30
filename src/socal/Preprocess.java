package socal;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Iterator;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.apache.poi.ss.usermodel.Cell;
import org.apache.poi.ss.usermodel.Row;
import org.apache.poi.xssf.usermodel.XSSFSheet;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;

public abstract class Preprocess implements StringUtils{
   private final String REGEX_WHITE_LIST = "[^((a-zA-Z'Ññ\"’\\s-)|([\\r\\n\\t]))]+";
   private ArrayList<String> stopwords   = new ArrayList<>();
   
   protected String outputPath;
   protected String[][] data;

   public Preprocess(String inputPath, String outputPath, String stopwordsPath) {
      try {
         this.outputPath = outputPath;
         setData(inputPath);
         setStopWords(stopwordsPath);
         
      } catch (IOException ex) {
         Logger.getLogger(Preprocess.class.getName()).log(Level.SEVERE, null, ex);
      }
   }
   
//<editor-fold defaultstate="collapsed" desc="Getters">

   public String[] getDataAtIndex(int index) {
      return this.data[index];
   }

   public String[][] getAllData() {
      return this.data;
   }

   public String[] getUniqueWords(int index) {
      return new HashSet<>(Arrays.asList(data[index])).toArray(new String[0]);
   }

   public String[] getUniqueWords() {
      List<String> listWords = new ArrayList<>();
      for (String[] dataGroup : data) {
         listWords.addAll(Arrays.asList(dataGroup));
      }
      String[] tempData = listWords.toArray(new String[listWords.size()]);

      return new HashSet<>(Arrays.asList(tempData)).toArray(new String[0]);
   }




//</editor-fold>
   
//<editor-fold defaultstate="collapsed" desc="Setters">
   private void setStopWords(String stopwordsPath) throws FileNotFoundException, IOException{
      BufferedReader br           = new BufferedReader(new FileReader(stopwordsPath));
      
      String line;
      while ((line = br.readLine()) != null) {
         stopwords.add(line.trim());
      }
   }

   private void setData(String path) throws FileNotFoundException, IOException {
      List<List<String>> tempList = new ArrayList<>();
      
      //Get the workbook instance for XLS file
      try (FileInputStream file = new FileInputStream(new File(path))) {
         //Get the workbook instance for XLS file
         XSSFWorkbook workbook = new XSSFWorkbook(file);
         
         //Get first sheet from the workbook
         XSSFSheet sheet = workbook.getSheetAt(0);
         
         //Iterate through each rows from first sheet
         Iterator<Row> rowIterator = sheet.iterator();
         while (rowIterator.hasNext()) {
            Row row = rowIterator.next();
            
            //For each row, iterate through each columns
            Iterator<Cell> cellIterator = row.cellIterator();
            while (cellIterator.hasNext()) {
               Cell cell = cellIterator.next();
               String contents = cell.getStringCellValue();
               
               if (!contents.isEmpty()) {
                  ArrayList<String> dataRow = new ArrayList<>(
                     Arrays.asList(cell.getStringCellValue()
                     .replaceAll(REGEX_WHITE_LIST, " ")
                     .replaceAll(" +", " ")
                     .split(" ")));
                  for (int i = 0; i < dataRow.size(); i++) {
                     dataRow.set(i, dataRow.get(i).toLowerCase());
                  }
                  tempList.add(dataRow);
               }
            }
         }
      }
      int size = tempList.size();
      data = new String[size][];
      
      int i = 0;
      for (List<String> list : tempList) {
         data[i++] = list.toArray(new String[list.size()]);
      }
   }
//</editor-fold>

//<editor-fold defaultstate="collapsed" desc="Outputs">
//   public abstract void csvOutput(double outputs) throws IOException;
//   
//   public abstract void excelOutput(double outputs) throws IOException;

   protected abstract void output(double outputs, boolean isExcel) throws IOException;   
//</editor-fold>
   
   protected void removeStopWords(Collection collection) {
      collection.removeAll(stopwords);
   }
   
   @Override
	public String format(String word){
		return word.toLowerCase().trim();
	}
}
