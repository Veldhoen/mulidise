import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


public class ApClassifyF1 {

    private static Map<String,String> getParamMap(String args[]) {
        HashMap<String,String> name2Value = new HashMap<String, String>();

        int idx = 0;
        while (idx < args.length) {
            String key = args[idx];
            if (!key.startsWith("--")) {
                return null;
            }
            idx++;
            if (idx == args.length) {
                return null;
            }
            name2Value.put(key, args[idx]);
            idx++;
        }
        return name2Value;

    }

    
    private static void printUsageExit() {
        System.out.println("Usage: java ApClassify --test-set [file name] --model-name [file name]");
    }
    /**
     * @param args
     */
    public static void main(String[] args) throws  IOException {
        Map<String,String> paramMap = getParamMap(args);
        if (paramMap.size() == 0) {
            printUsageExit();
        }

        System.out.print("Params: " + paramMap + "  ");


        String fileName = paramMap.get("--test-set");
        String modelName = paramMap.get("--model-name");

        InputReader reader = new InputReader();
        List<Example> data = reader.readData(fileName);

        Model mod = Model.load(modelName);

        int correct = 0;
        double tP = 0;
        double fP = 0;
        double tN = 0;
        double fN = 0;

        for (Example ex : data) {
            int pred = mod.predictLabel(ex.sv);
//	    System.out.println("prediction: "+pred+", true:"+ex.label);

            if (pred == 1) {
              // predicted positive (1)
              if (pred == ex.label) tP ++;
              else fP ++;
            }
            else {
              // predicted negative (2)
              if (pred == ex.label) tN ++;
              else fN ++;
            }

            if (pred == ex.label) {
                correct++;
            }

        }
        double precision=tP/(tP+fP);
        double recall = tP/(tP+fN);
        double f1 = 2*precision*recall/(precision+recall);
        System.out.println("Accuracy: " + (((double) correct) / data.size())+", precision: " + precision+", recall: "+recall+", F1: "+f1);

    }

}
