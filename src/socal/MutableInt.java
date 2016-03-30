package socal;

public class MutableInt implements Comparable<MutableInt>{
   private int value = 1; //We start at 1 since we're counting
   
   public void increment(){
      ++value;
   }
   
   public int get(){
      return value;
   }

   @Override
   public int compareTo(MutableInt that) {
      return that.get()-this.get();
   }

   @Override
   public String toString(){
      return Integer.toString(value);
   }
}