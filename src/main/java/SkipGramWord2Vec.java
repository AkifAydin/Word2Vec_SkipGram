
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

  public static final String TARGET = "day";

  // Source:
  // https://deeplearning4j.konduit.ai/v/en-1.0.0-beta7/language-processing/word2vec#just-give-me-the-code

  public static void main(String[] args) throws Exception {

    // Gets Path to Text file
    String filePath = "raw_sentences.txt";

    // Iterator to Iterate over a dataset.
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
            .seed(42) // random seed
            .windowSize(5)  // size of window
            .iterate(iter) // tells the net what batch of the dataset its training on
            .tokenizerFactory(t)  // the token generator
            .build();

    System.out.println("Fitting Word2Vec model....");
    vec.fit();  // tells the configured net to begin training

    System.out.println("Writing word vectors to text file....");

    // Write word vectors to file (deprecated but does exactly what we want)
    WordVectorSerializer.writeWordVectors(vec, new File("wordsVector.txt"));
    // WordVectorSerializer.writeWord2VecModel(vec, new File("wordsVector.txt"));

    // Prints out the closest 10 words to "day"
    System.out.println("Closest Words of: " + TARGET);
    Collection<String> lst = vec.wordsNearest(TARGET, 10); //Target word
    System.out.println("10 Words closest to '" + TARGET + "': " + lst);

    // Words similarity score
    System.out.println("Word similarity score for '" + TARGET + "' and results:");
    for (String compWord : lst) {
      double simScore = vec.similarity(TARGET, compWord);
      System.out.println("\t" + compWord + ": " + simScore);
    }
  }
}
