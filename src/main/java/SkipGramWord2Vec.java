
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;

import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;

import java.io.File;
import java.util.Collection;

public class SkipGramWord2Vec {

  public static final String INPUT = "day";

  // Source:
  // https://deeplearning4j.konduit.ai/v/en-1.0.0-beta7/language-processing/word2vec#just-give-me-the-code

  public static void main(String[] args) throws Exception {

    // get path to Text file
    String filePath = "raw_sentences.txt";

    // Iterator to iterate over the dataset.
    // SentenceIterator returns strings.
    SentenceIterator iter = new BasicLineIterator(filePath);


    // Used in tokenizing the text.
    // To tokenize a text is to break it up into its atomic units,
    // creating a new token each time you hit a white space, for example.
    // A Sentence is represented as a series of tokens.
    TokenizerFactory t = new DefaultTokenizerFactory();
    t.setTokenPreProcessor(new CommonPreprocessor());

    System.out.println("Building model...");
    Word2Vec vec = new Word2Vec.Builder()
            .minWordFrequency(5) // only learn words appearing at least 5 times
            .layerSize(100) // specifies the number of features in the word vector
            .seed(42) // seed for random NN initialization
            .iterate(iter) // tells the net what batch of the dataset its training on
            .tokenizerFactory(t)  // feeds the builder the individual words of each batch
            .build();

    System.out.println("Fitting Word2Vec model....");
    vec.fit();  // tells the configured net to begin training

    System.out.println("Writing word vectors to text file....");
    // write word vectors to file (deprecated but does exactly what we want)
    WordVectorSerializer.writeWordVectors(vec, new File("wordsVector.txt"));
    // WordVectorSerializer.writeWord2VecModel(vec, new File("wordsVector.txt"));

    // prints out the closest 10 words to the input word
    Collection<String> lst = vec.wordsNearest(INPUT, 10);
    System.out.println("10 words closest to '" + INPUT + "': " + lst);

    // word similarity scores
    System.out.println("Word similarity score for '" + INPUT + "' and results:");
    for (String compWord : lst) {
      double simScore = vec.similarity(INPUT, compWord);
      System.out.println("\t" + compWord + ": " + simScore);
    }
  }
}
