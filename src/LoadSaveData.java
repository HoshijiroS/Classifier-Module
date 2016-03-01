import javax.swing.JFileChooser;
import javax.swing.filechooser.FileNameExtensionFilter;

public class LoadSaveData {
   String path;

   public String LoadData() {
      JFileChooser chooser = new JFileChooser();
      FileNameExtensionFilter filter = new FileNameExtensionFilter("ARFF files", "arff");
      chooser.setFileFilter(filter);
      int returnVal = chooser.showOpenDialog(null);
      if (returnVal == JFileChooser.APPROVE_OPTION) {
         System.out.println("You chose to open this file: " + chooser.getSelectedFile().getName());
         path = chooser.getSelectedFile().getAbsolutePath();
      }

      return path;
   }
}
