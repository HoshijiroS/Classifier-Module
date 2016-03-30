import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.InputMismatchException;

import edu.stanford.nlp.tagger.maxent.MaxentTagger;
import socal.PreprocessSO_CAL;
import socal.Weight;


public class SOCAL extends RuleBasedClassifier{
  private final String WEIGHT_PATH = "weights/";
  private ArrayList<Weight> weights;

  public SOCAL(ArrayList<String> articles, String[] sentiments) {
    super(articles, sentiments);
  }
  
  public SOCAL(ArrayList<String> articles, ArrayList<String> sentiments) {
    super(articles, sentiments);
  }

  @Override
  public void preprocessArticles() {
    final String QUOTES = "(\"[^\"]+\")|(\\(([^)]+)\\))";
    final String SPACES = "\\s+";
   
    try{
      /*
       * Remove text within quotes
       * Change multiple whitespaces to single space
       */
      for(int i = 0 ; i < articles.size(); i++){
        String article = articles.get(i);
        if(article.startsWith("\"") && article.endsWith("\""))
              article = article.substring(1, article.length()-1);
          article = article.replaceAll(QUOTES, "");
          article = article.replaceAll(SPACES, " ");
          articles.set(i, article);
      }
    }catch(Exception e){
      e.printStackTrace();
    }    
    
  }

  @Override
  public void classify() {
    initializeWeights();
    MaxentTagger tagger = new MaxentTagger("tagger/filipino.tagger");
    
    if(articles.size()!=classifierPredictions.length){
      throw new InputMismatchException("Articles size ("+articles.size()+") != classifierPredictions.length ("+classifierPredictions.length+")");
    }
    
    for(int i=0;i<articles.size();i++){
      int nWeight = 0;
      String taggedArticle = tagger.tagString(articles.get(i));
      
      for(String s : taggedArticle.split(" ")){
        String taggedWord[] = s.split("/");
        if(taggedWord.length==2){
          nWeight+=getWeight(taggedWord[0], taggedWord[1].charAt(0));
        }
      }
      
      String classification;
      if(nWeight > 0){
        classification = POSITIVE;
      } else if (nWeight <0){
        classification = NEGATIVE;
      } else{
        classification = NEUTRAL;
      }
      classifierPredictions[i]=classification;
      System.out.println("");
    }
    
    /*
     * insert code that would solve for sentiments
     */
    
    
//
//    String inputPath     = "Input/Articles.xlsx";
//    String outputPath    = "src/Output/";
//    String stopwordsPath = "funcwordsfil.txt";
//    PreprocessSO_CAL socal = new PreprocessSO_CAL(inputPath, outputPath, stopwordsPath);
    
    
    //BY HERE
    //classifierPredictions have been populated
    
  //Solve for accuracy
    int match = 0;
    for(int i = 0 ; i < actualPredictions.length; i++){
      if(actualPredictions[i].equalsIgnoreCase(classifierPredictions[i]))
        match++;
    }
    
    accuracy = (match / (actualPredictions.length * 1.0)) * 100;
    System.out.println(accuracy);
  }
  
  private int getWeight(String s, char tag){
    for(Weight weight : weights){
      if(weight.getTag() == tag){
        return weight.getWordValue(s);
      }
    }
    return 0;
  }
  
  private void initializeWeights() {
    this.weights = new ArrayList<>();
    weights.add(new Weight(WEIGHT_PATH + "ADJ.xlsx", 'J'));
    weights.add(new Weight(WEIGHT_PATH + "ADV.xlsx", 'R'));
//    weights.add(new Weight(WEIGHT_PATH+"INT.xlsx", '?'));
    weights.add(new Weight(WEIGHT_PATH + "NOUN.xlsx", 'N'));
    weights.add(new Weight(WEIGHT_PATH + "VERB.xlsx", 'V'));
 }
}
