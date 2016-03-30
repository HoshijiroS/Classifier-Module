package socal;

public class TaggedWords {
   private final String word;
   private final char tag;

   public TaggedWords(String wordWithTag) {
      String[] temp = wordWithTag.split("/");
      
      if(temp.length==2){
         tag  = temp[1].charAt(0);
         word = temp[0];
      } else{
         word = "___";
         tag = '?';
      }
   }

   public String getWord() {
      return word;
   }

   public char getTag() {
      return tag;
   }

   @Override
   public String toString() {
      return String.format("%s: %c: ", word, tag);
   }
}
