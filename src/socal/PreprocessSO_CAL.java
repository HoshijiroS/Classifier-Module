package socal;

import edu.stanford.nlp.tagger.maxent.MaxentTagger;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;

public class PreprocessSO_CAL extends Preprocess {
   private final String WEIGHT_PATH = "weights\\";
   private final String TAGGER_PATH = "tagger\\filipino.tagger";
   private final String SPACE = " ";
   private final String[] articleSentiments;
   private final MaxentTagger tagger;

   private ArrayList<Weight> weights;

   public PreprocessSO_CAL(String inputPath, String outputPath, String stopwordsPath) {
      super(inputPath, outputPath, stopwordsPath);
      initializeWeights();

      tagger = new MaxentTagger(TAGGER_PATH);
      int size = data.length;
      articleSentiments = new String[size];

      for (int i = 0; i < size; i++) {
         tagArticle(data[i], 0);
      }

      System.out.println(Arrays.toString(articleSentiments));
   }

   private void tagArticle(String[] toTag, int index) {
      String[] words = tagger.tagString(inputToString(toTag)).split(SPACE);
      
      TaggedWords[] taggedWords = setTaggedWords(words);

      int articleWeight = getArticleWeight(taggedWords);

      System.out.println("Weight is " + articleWeight);
      String sentiment;
      if (articleWeight > 0) {
         sentiment = "Positive";
      } else if (articleWeight < 0) {
         sentiment = "Negative";
      } else {
         sentiment = "Neutral";
      }
      articleSentiments[index] = sentiment;
   }

   private TaggedWords[] setTaggedWords(String[] words) {
      TaggedWords[] taggedWords = new TaggedWords[words.length];
      int i = 0;
      for (String word : words) {
         taggedWords[i++] = new TaggedWords(word);
      }
      return taggedWords;
   }

   private int getArticleWeight(TaggedWords[] taggedWords) {
      int articleWeight = 0;
      for (Weight w : weights) {
         articleWeight += Arrays.stream(taggedWords)
            .filter(tw->tw.getTag() == w.getTag()) //The Weight tag is equal to the tag of the current word
            .map(tw->w.getWordValue(tw.getWord())) //Get only the weights of each word (int)
            .reduce(0, (acc, item)->acc + item);   //Collect all the weights, and return their total
      }
      return articleWeight;
   }

   private String inputToString(String[] toTag) {
      StringBuilder sb = new StringBuilder();
      for (String s : toTag) {
         sb.append(format(s));
         sb.append(SPACE);
      }
      return sb.toString();
   }

   public void excelOutput(double outputs) throws IOException {
      output(outputs, true);
   }

   @Override
   protected void output(double outputs, boolean isExcel) throws IOException {
      for(int i=0;i<articleSentiments.length;i++){
         System.out.println(i+"\t"+articleSentiments[i]);
      }
//      if (isExcel) {
//         ExcelTools.makeExcelOutput(data, keys, outputPath + percentage + "%.xslx");
//      } else {
//         CSVTools.makeCSVOutput(data, keys, outputPath + percentage + "%.csv");
//      }
   }

   private void initializeWeights() {
      this.weights = new ArrayList<>();
      weights.add(new Weight(WEIGHT_PATH + "ADJ.xlsx", 'J'));
      weights.add(new Weight(WEIGHT_PATH + "ADV.xlsx", 'R'));
//      weights.add(new Weight(WEIGHT_PATH+"INT.xlsx", '?'));
      weights.add(new Weight(WEIGHT_PATH + "NOUN.xlsx", 'N'));
      weights.add(new Weight(WEIGHT_PATH + "VERB.xlsx", 'V'));
   }
   
   public String[] getArticleSentiments(){
     return this.articleSentiments;
   }
}
