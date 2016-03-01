import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;

import javax.swing.JFileChooser;
import javax.swing.filechooser.FileNameExtensionFilter;

public class FileHandler {
   public BufferedReader readFile() {
      String filename = null;
      JFileChooser chooser = new JFileChooser();
      FileNameExtensionFilter filter = new FileNameExtensionFilter("ARFF files", "arff");
      chooser.setFileFilter(filter);
      int returnVal = chooser.showOpenDialog(null);
      if (returnVal == JFileChooser.APPROVE_OPTION) {
         System.out.println("You chose to open this file: " + chooser.getSelectedFile().getName());
         filename = chooser.getSelectedFile().getAbsolutePath();
      }

      BufferedReader inputReader = null;

      try {
         inputReader = new BufferedReader(new FileReader(filename));
      } catch (FileNotFoundException ex) {
         System.err.println("File not found: " + filename);
      }

      return inputReader;
   }
}
