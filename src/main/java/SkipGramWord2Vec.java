import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

import java.io.File;
import java.util.Collection;

/**
 * Created by afjal on 12/3/16.
 */
public class SkipGramWord2Vec {

    public static void main(String[] args) throws Exception {

        // Gets Path to Text file
        String filePath = "raw_sentences.txt";

        SentenceIterator iter = new BasicLineIterator(filePath);
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        System.out.println("Building model....");
        Word2Vec vec = new Word2Vec.Builder()
                .minWordFrequency(5)
                .iterations(1)
                .layerSize(100)
                .seed(42)
                .windowSize(5)
                .iterate(iter)
                .tokenizerFactory(t)
                .build();

        System.out.println("Fitting Word2Vec model....");
        vec.fit();

        System.out.println("Writing word vectors to text file....");

        // Write word vectors to file
        WordVectorSerializer.writeWordVectors(vec, new File("wordsVector.txt"));

        // Prints out the closest 10 words to "day".
        System.out.println("Closest Words of day");
        Collection<String> lst = vec.wordsNearest("day", 10);
        System.out.println("10 Words closest to 'day': " + lst);

        // Words similarity...
        String w1="people",w2="national";
        double simScore=vec.similarity(w1,w2);
        System.out.println(w1+" and "+w2+" word similarity score: "+simScore);

    }
}
