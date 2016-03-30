import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;

public class ArticleFrequency {

    private double posProb;
    private double negProb;
    private double neutProb;
    private String finalSentiment;

    public ArticleFrequency(double posProb, double negProb, double neutProb) {
        this.posProb = posProb;
        this.negProb = negProb;
        this.neutProb = neutProb;
        
        if(posProb > negProb && posProb > neutProb)
            this.finalSentiment = "Positive";
        else if(negProb > posProb && negProb > neutProb)
            this.finalSentiment = "Negative";
        else if(neutProb > posProb && neutProb > negProb)
            this.finalSentiment = "Neutral";
        else
            this.finalSentiment = "TIE";
    }
    

    public double getPosProb() {
        return posProb;
    }

    public void setPosProb(double posProb) {
        this.posProb = posProb;
    }

    public double getNegProb() {
        return negProb;
    }

    public void setNegProb(double negProb) {
        this.negProb = negProb;
    }

    public double getNeutProb() {
        return neutProb;
    }

    public void setNeutProb(double neutProb) {
        this.neutProb = neutProb;
    }
    
    public String getFinalSentiment(){
        return this.finalSentiment;
    }
    
}
