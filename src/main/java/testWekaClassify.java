import java.io.File;
import java.util.logging.Logger;

public class testWekaClassify {

    /**
     * Main method. With an example usage of this class.
     */

    public static void main(String[] args) throws Exception {

        Logger LOGGER = Logger.getLogger("DebitCreditService");


        DebitCreditWekaClassifier wt = new DebitCreditWekaClassifier();
        // URL url = wt.getClass().getClassLoader().getResource("./");

//        final String MODEL = url.getPath() + "model.dat";

        //TODO Change debit-credit model to default credit direction

        final String MODEL = "model/income_outcome_model.dat";

        if (new File(MODEL).exists()) {
            wt.loadModel(MODEL);
        } else {
            wt.transform();
            wt.fit();
            wt.saveModel(MODEL);
        }

        LOGGER.warning("\nTest debit credit weka classifyer\n");
        //run test predictions
        LOGGER.info("text 'bought' is " + wt.predict("bought a chicken ?"));
        LOGGER.info("text 'spend all my money' is " + wt.predict("spend all my money"));
        //TO DO some test...

        //run evaluation
        LOGGER.info("Payment" + " " + wt.predict("Payment"));
        LOGGER.info("Payment" + " " + wt.predict("Payment"));
        LOGGER.info("bought a bun" + " " + wt.predict("bought a bun"));
        LOGGER.info("buy nuts" + " " + wt.predict("buy nuts"));
        LOGGER.info("pay pall" + " " + wt.predict("pay pall"));
        LOGGER.info("salary" + " " + wt.predict("salary"));
        LOGGER.info("profit" + " " + wt.predict("salary"));
        LOGGER.info("spend" + " " + wt.predict("spend"));


        LOGGER.info("Evaluation Result: \n"+wt.evaluate());
        //TODO Add complex test in model


    }

}
