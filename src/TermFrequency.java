import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;


public class TermFrequency extends RuleBasedClassifier{
  
  public TermFrequency(ArrayList<String> articles, String[] sentiments){
    super(articles, sentiments);
  }
  
  public TermFrequency(ArrayList<String> articles, ArrayList<String> sentiments){
    super(articles, sentiments);
  }
  
  @Override
  public void preprocessArticles() {
    final String QUOTES = "(\"[^\"]+\")|(\\(([^)]+)\\))";
    final String SPACES = "\\s+";
    ArrayList<String> list = new ArrayList<String>();
    ArrayList<String> stopWords = new ArrayList<String>();
   
    try {
        //STOPWORDS
        //FIL
        BufferedReader br = new BufferedReader(new FileReader("funcwordsfil.txt"));
        String word;
        while((word = br.readLine()) != null)
            stopWords.add(word.trim().toLowerCase());
        //ENG
        br = new BufferedReader(new FileReader("funcwordseng.txt"));
        while((word = br.readLine()) != null)
            stopWords.add(word.trim().toLowerCase());
        br.close();
    } catch (Exception ex) {
        ex.printStackTrace();
    }
    try{
      /*
       * Remove text within quotes
       * Change multiple whitespaces to single space
       */
      for(String article : articles){
        if(article.startsWith("\"") && article.endsWith("\""))
              article = article.substring(1, article.length()-1);
          article = article.replaceAll(QUOTES, "");
          article = article.replaceAll(SPACES, " ");
          list.add(article);
      }
      /*
       * Remove stop words
       */
      for(int i = 0 ; i < list.size(); i++){
         String content = list.get(i);
         content = content.toLowerCase();
        
         List<String> words = new LinkedList<>(Arrays.asList(content.split(" ")));
           words.removeAll(stopWords);
           content = "";
           for (String str : words)
               content += str + " ";
           if(content.length() > 1)
               content.substring(0, content.length()-1);
           articles.set(i, content);
      }
    }catch(Exception e){
      e.printStackTrace();
    }    
  }
  
  @Override
  public void classify() {

    HashMap<String, Integer> posHash = new HashMap<String, Integer>();
    HashMap<String, Integer> negHash = new HashMap<String, Integer>();   
    HashMap<String, Integer> neutHash = new HashMap<String, Integer>();
    ArrayList<Article> articlesArr = new ArrayList<Article>();
    
    for(int i = 0 ; i < this.articles.size(); i++){
      String article = this.articles.get(i);
      String sentiment = this.actualPredictions[i] ;
      Article curr = new Article(article, sentiment);
      articlesArr.add(curr);
      
      switch(sentiment.toUpperCase()){
          case "POSITIVE":
              posHash = curr.addToHash(posHash);
              break;
          case "NEGATIVE":
              negHash = curr.addToHash(negHash);
              break;
          case "NEUTRAL":
              neutHash = curr.addToHash(neutHash);
              break;
          default: System.out.println("UNKNOWN SENTIMENT");
              break;
      }
    }
    
    ArrayList<ArticleFrequency> articleFrequencies = new ArrayList<ArticleFrequency>();
    
    for(Article article : articlesArr){
       ArrayList<String> wordList = article.getWords();
       
        double negTotal = 0.0;
        double posTotal = 0.0;
        double neutTotal = 0.0;
        for(String word : wordList){
           int negFreq = 0; 
           int posFreq = 0; 
           int neutFreq = 0;
           
           double termFreq = 0.0;
           
           if(posHash.containsKey(word))
               posFreq = posHash.get(word);
           if(negHash.containsKey(word))
               negFreq = negHash.get(word);
           if(neutHash.containsKey(word))
               neutFreq = neutHash.get(word);
           
           
           switch(article.getSentiment().toUpperCase()){
               case "POSITIVE":
                   if(posHash.containsKey(word)){
                       posFreq--;
                   }
                   break;
               case "NEGATIVE":
                   if(negHash.containsKey(word)){
                       negFreq--;
                   }
                   break;
               case "NEUTRAL":
                   if(neutHash.containsKey(word)){
                       neutFreq--;
                   }
                   break;
               default: System.out.println("INVALID SENTIMENT FREQUENCY");
                   break;
           }
           
           int totalFreq = posFreq + negFreq + neutFreq;
           
           if(totalFreq > 0){
                posTotal += (posFreq / totalFreq);
                negTotal += (negFreq / totalFreq);
                neutTotal += (neutFreq / totalFreq);
           }
       }
        articleFrequencies.add(new ArticleFrequency(posTotal, negTotal, neutTotal));
   }
    
    for(int i = 0 ; i < articleFrequencies.size(); i++){
      ArticleFrequency af = articleFrequencies.get(i);
      this.classifierPredictions[i] = af.getFinalSentiment();
    }
    
    //Solve for accuracy
    int match = 0;
    for(int i = 0 ; i < actualPredictions.length; i++){
      if(actualPredictions[i].equalsIgnoreCase(classifierPredictions[i]))
        match++;
    }
    
    accuracy = (match / (actualPredictions.length * 1.0)) * 100;
    System.out.println(accuracy);
  }

}


