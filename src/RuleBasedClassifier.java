import java.util.ArrayList;


public abstract class RuleBasedClassifier {
  protected ArrayList<String> articles;
  protected String[] actualPredictions;
  protected String[] classifierPredictions;
  protected double accuracy;
  
  public abstract void classify();
  public abstract void preprocessArticles();
  
  protected final String POSITIVE = "Positive";
  protected final String NEGATIVE = "Negative";
  protected final String NEUTRAL  = "Neutral";
  
  public RuleBasedClassifier(ArrayList<String> articles, String[] sentiments){
    this.articles = articles;
    this.actualPredictions = sentiments;
    this.classifierPredictions = new String[actualPredictions.length];
    preprocessArticles();
    classify();    
  }
  
  public RuleBasedClassifier(ArrayList<String> articles, ArrayList<String> sentiments){
    this.articles = articles;
    this.actualPredictions = sentiments.toArray(new String[sentiments.size()]);
    this.classifierPredictions = new String[actualPredictions.length];
    preprocessArticles();
    classify();
  }

  public String[] getActualPredictions() {
    return actualPredictions;
  }

  public void setActualPredictions(String[] actualPredictions) {
    this.actualPredictions = actualPredictions;
  }
  
  public String[] getClassifierPredictions(){
    return classifierPredictions;
  }

  public double getAccuracy() {
    return accuracy;
  }

  public void setAccuracy(double accuracy) {
    this.accuracy = accuracy;
  }
  
}
