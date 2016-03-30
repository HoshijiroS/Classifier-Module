import java.util.ArrayList;
import java.util.HashMap;


public class Article {
    private String articleContent;
    private ArrayList<String> words;
    private String sentiment;
    
    public Article(String articleContent, ArrayList<String> words, String sentiment){
        this.articleContent = articleContent;
        this.words = words;
        this.sentiment = sentiment;
    }
    
    public Article(String articleContent, String sentiment){
        this.articleContent = articleContent;
        this.words = new ArrayList<String>();
        this.sentiment = sentiment;
        this.getListOfWords();
    }
    
    public void getListOfWords(){
        for(String word : articleContent.split(" ")){
            word = word.toLowerCase();
            if(!this.words.contains(word))
                this.words.add(word);
        }
    }

    public HashMap<String, Integer> addToHash(HashMap<String, Integer> hash){
        for(String word : words){
            if(hash.containsKey(word))
                hash.put(word, (int)hash.get(word) + 1);
            else
                hash.put(word, 1);
        }
        return hash;
    }
    
    public String getArticleContent() {
        return articleContent;
    }

    public void setArticleContent(String articleContent) {
        this.articleContent = articleContent;
    }

    public ArrayList<String> getWords() {
        return words;
    }

    public void setWords(ArrayList<String> words) {
        this.words = words;
    }

    public String getSentiment() {
        return sentiment;
    }

    public void setSentiment(String sentiment) {
        this.sentiment = sentiment;
    }
    
    
}
