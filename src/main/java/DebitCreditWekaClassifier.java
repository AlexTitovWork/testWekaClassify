import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;
import weka.core.converters.ArffSaver;
import weka.core.tokenizers.NGramTokenizer;
import weka.filters.unsupervised.attribute.StringToWordVector;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;

// import java.net.URL;


/**
 * DEBIT CREDIT classifier based  Weka libs, for detect transaction direction.
 *
 * @author Alex Titov https://github.com/AlexTitovWork/testWekaClassify
 * Thanks a lot for original SPAM filter code Alfred Francis!
 * @see https://github.com/alfredfrancis/spam-classification-weka-java/blob/master/WekaClassifier.java
 * @see http://https://waikato.github.io/weka-wiki/use_weka_in_your_java_code/
 * @see https://www.cs.waikato.ac.nz/ml/index.html
 */
public class DebitCreditWekaClassifier {

    private static Logger LOGGER = Logger.getLogger("DebitCreditInternalSystem");

    private FilteredClassifier classifier;

    //declare train and test data Instances
    private Instances trainData;


    //declare attributes of Instance
    private ArrayList<Attribute> wekaAttributes;


    private static final String TRAIN_DATA = "dataset/train_income_outcome.txt";
    private static final String TRAIN_ARFF_ARFF = "dataset/train_income_outcome.arff";
    private static final String TEST_DATA = "dataset/test_income_outcome.txt";
    private static final String TEST_DATA_ARFF = "dataset/test_income_outcome.arff";


    DebitCreditWekaClassifier() {

        /*
         * Class for running an arbitrary classifier on data that has been passed through an arbitrary filter.
         * Like the classifier, the structure of the filter is based exclusively on the training data and test
         * instances will be processed by the filter without changing their structure.
         * If unequal instance weights or attribute weights are present,
         * and the filter or the classifier are unable to deal with them,
         * the instances and/or attributes are resampled with replacement
         * based on the weights before they are passed to the filter or the classifier (as appropriate).
         */
        classifier = new FilteredClassifier();

        /**
         * Class for building and using a multinomial Naive Bayes classifier. For more information see,
         *
         * Andrew Mccallum, Kamal Nigam: A Comparison of Event Models for Naive Bayes Text Classification.
         * In: AAAI-98 Workshop on 'Learning for Text Categorization', 1998.
         * https://weka.sourceforge.io/doc.dev/weka/classifiers/bayes/NaiveBayesMultinomial.html
         */
        classifier.setClassifier(new NaiveBayesMultinomial());

        // Declare text attribute to hold the message
        Attribute attributeText = new Attribute("text", (List<String>) null);

        /**
         * Declare the label attribute along with its values
          */
        ArrayList<String> classAttributeValues = new ArrayList<>();
        classAttributeValues.add("debit");
        classAttributeValues.add("credit");
        Attribute classAttribute = new Attribute("label", classAttributeValues);

        /**
         * Built the feature vector "wekaAttributes"
         */
        wekaAttributes = new ArrayList<>();
        wekaAttributes.add(classAttribute);
        wekaAttributes.add(attributeText);

    }

    /**
     * Load training data and set feature generators
     */
    public void transform() {
        try {
            trainData = loadDataset(TRAIN_DATA);
            saveArff(trainData, TRAIN_ARFF_ARFF);
            /**
             * Сreate the filter
             * and set the attribute to be transformed from text into a feature vector (the last one)
             */
            StringToWordVector filter = new StringToWordVector();
            /**
             * https://waikato.github.io/weka-wiki/adding_attributes_to_dataset/
             */
            filter.setAttributeIndices("last");
            /**
             * Add ngram tokenizer to filter with min and max length set to 1
             */
            NGramTokenizer tokenizer = new NGramTokenizer();
            tokenizer.setNGramMinSize(1);
            tokenizer.setNGramMaxSize(1);
            /**
             * Tokenize based on delimiter
             * not alphanumeric regexp "\W"
             */
            tokenizer.setDelimiters("\\W");
            filter.setTokenizer(tokenizer);
            /**
             * To lowercase converting,
             * as standard filter's procedure.
             */
            filter.setLowerCaseTokens(true);
            /**
             *  Set filter to classifier
             */
            classifier.setFilter(filter);
        } catch (Exception e) {
            LOGGER.warning(e.getMessage());
        }
    }

    /**
     * Build prepared classifier on the training data
     */
    public void fit() {
        try {
            classifier.buildClassifier(trainData);
        } catch (Exception e) {
            LOGGER.warning(e.getMessage());
        }
    }


    /**
     * classify a new message into income or outcome.
     *
     * @param text to be classified.
     * @return a class label (income or outcome )
     */
    public String predict(String text) {
        try {
            // create new Instance for prediction.
            DenseInstance newinstance = new DenseInstance(2);

            //weka demand a dataset to be set to new Instance
            Instances newDataset = new Instances("predictiondata", wekaAttributes, 1);
            newDataset.setClassIndex(0);

            newinstance.setDataset(newDataset);

            // text attribute value set to value to be predicted
            newinstance.setValue(wekaAttributes.get(1), text);

            // predict most likely class for the instance
            double pred = classifier.classifyInstance(newinstance);

            // return original label
            return newDataset.classAttribute().value((int) pred);
        } catch (Exception e) {
            LOGGER.warning(e.getMessage());
            return null;
        }
    }

    /**
     * evaluate the classifier with the Test data
     *
     * @return evaluation summary as string
     */
    public String evaluate() {
        try {
            //load testdata
            Instances testData;
            if (new File(TEST_DATA_ARFF).exists()) {
                testData = loadArff(TEST_DATA_ARFF);
                testData.setClassIndex(0);
            } else {
                testData = loadDataset(TEST_DATA);
                saveArff(testData, TEST_DATA_ARFF);
            }

            Evaluation eval = new Evaluation(testData);
            eval.evaluateModel(classifier, testData);
            return eval.toSummaryString();
        } catch (Exception e) {
            LOGGER.warning(e.getMessage());
            return null;
        }
    }

    /**
     * Model loader
     *
     * @param filename The name of the file that stores the text.
     */
    public void loadModel(String filename) {
        try {
            ObjectInputStream in = new ObjectInputStream(new FileInputStream(filename));
            Object tmp = in.readObject();
            classifier = (FilteredClassifier) tmp;
            in.close();
            LOGGER.info("Model successfully loaded: " + filename);
        } catch (FileNotFoundException e) {
            LOGGER.warning(e.getMessage());
        } catch (IOException e) {
            LOGGER.warning(e.getMessage());
        } catch (ClassNotFoundException e) {
            LOGGER.warning(e.getMessage());
        }
    }

    /**
     * This method saves the trained model into a file. This is done by
     * simple serialization of the classifier object.
     *
     * @param filename The name of the file that will store the trained model.
     */

    public void saveModel(String filename) {
        try {
            ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(filename));
            out.writeObject(classifier);
            out.close();
            LOGGER.info("Saved model: " + filename);
        } catch (IOException e) {
            LOGGER.warning(e.getMessage());
        }
    }

    /**
     * Loads a data set into a text file, separated by spaces, and converts it to Arff format
     * Attribute-Relation File Format (ARFF)
     * https://www.cs.waikato.ac.nz/ml/weka/arff.html
     *
     * @param filename
     * @return Instances of ARFF file
     */
    public Instances loadDataset(String filename) {
        /*
         *  Create an empty training set
         *  name the relation “Rel”.
         *  set intial capacity of 8*
         */

        Instances dataset = new Instances("Debit Credit", wekaAttributes, 8);

        // Set class index
        dataset.setClassIndex(0);

        /**
         * Read data file, parse text and add to instance
         */
        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            for (String line;
                 (line = br.readLine()) != null; ) {
                // split at first occurance of n no. of words
                String[] parts = line.split("\\s+", 2);

                // basic validation
                if (!parts[0].isEmpty() && !parts[1].isEmpty()) {

                    DenseInstance row = new DenseInstance(2);
                    row.setValue(wekaAttributes.get(0), parts[0]);
                    row.setValue(wekaAttributes.get(1), parts[1]);

                    // add row to instances
                    dataset.add(row);
                }

            }

        } catch (IOException e) {
            LOGGER.warning(e.getMessage());
        } catch (ArrayIndexOutOfBoundsException e) {
            LOGGER.info("Bad row detected!.");
        }
        return dataset;

    }

    /**
     * Loads a dataset in ARFF format. If the file does not exist, or
     * it has a wrong format, the attribute trainData is null.
     *
     * @param filename The name of the file that stores the dataset.
     */
    public Instances loadArff(String filename) {
        try {
            BufferedReader reader = new BufferedReader(new FileReader(filename));
            ArffReader arff = new ArffReader(reader);
            Instances dataset = arff.getData();
            // replace with logger System.out.println("loaded dataset: " + fileName);
            reader.close();
            return dataset;
        } catch (IOException e) {
            LOGGER.warning(e.getMessage());
            return null;
        }
    }

    /**
     * This method saves a dataset in ARFF format.
     * Attribute-Relation File Format (ARFF)
     *
     * @param dataset  dataset in arff format
     * @param filename The name of the file that stores the dataset.
     */
    public void saveArff(Instances dataset, String filename) {
        try {
            // initialize 
            ArffSaver arffSaverInstance = new ArffSaver();
            arffSaverInstance.setInstances(dataset);
            arffSaverInstance.setFile(new File(filename));
            arffSaverInstance.writeBatch();
        } catch (IOException e) {
            LOGGER.warning(e.getMessage());
        }
    }

}

