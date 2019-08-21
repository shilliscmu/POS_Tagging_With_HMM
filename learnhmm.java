import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;

public class learnhmm {

    public static void main(String[] args) {
        // write your code here
        String trainInput = args[0];
        String indexToWord = args[1];
        String indexToTag = args[2];
        String hmmPrior = args[3];
        String hmmEmit = args[4];
        String hmmTrans = args[5];

        ArrayList<Double> initialProbsPi = new ArrayList<>();
        ArrayList<ArrayList<Double>> transitionProbsA = new ArrayList<>();
        ArrayList<ArrayList<Double>> emissionProbsB = new ArrayList<>();

        //Xs
        ArrayList<String> indexToWordData = readIndexData(indexToWord);
        //Ys
        ArrayList<String> indexToTagData = readIndexData(indexToTag);
        ArrayList<ArrayList<Pair<Integer, Integer>>> inputData = readInput(trainInput, indexToWordData, indexToTagData);

        initialProbsPi = computeInitialProbs(inputData, indexToTagData);
        printInitialProbs(hmmPrior, initialProbsPi);

        transitionProbsA = computeTransitionProbs(inputData, indexToTagData);
        printTransitionOrEmission(hmmTrans, transitionProbsA);

        emissionProbsB = computeEmissionProbs(inputData, indexToTagData, indexToWordData);
        printTransitionOrEmission(hmmEmit, emissionProbsB);

    }

    private static ArrayList<ArrayList<Double>> computeEmissionProbs(ArrayList<ArrayList<Pair<Integer, Integer>>> inputData, ArrayList<String> indexToTagData, ArrayList<String> indexToWordData) {
        ArrayList<ArrayList<Double>> toReturn = new ArrayList<>();
        HashMap<Pair<Integer, Integer>, Integer> wordStateCount = new HashMap<>();
        ArrayList<Double> denoms = new ArrayList<>();
        for(ArrayList<Pair<Integer, Integer>> sentence : inputData) {
            for(Pair<Integer, Integer> wordStatePair : sentence) {
                int j = wordStatePair.getValue();
                int k = wordStatePair.getKey();
                wordStateCount.merge(new Pair(j, k), 1, Integer::sum);
            }
        }
        for(int j = 0; j < indexToTagData.size(); j++) {
            double denom = 0;
            for(int k = 0; k < indexToWordData.size(); k++) {
                if(wordStateCount.containsKey(new Pair(j,k))) {
                    denom += (wordStateCount.get(new Pair(j, k)));
                }
                denom += 1;
            }
            denoms.add(denom);
        }

        for(int j = 0; j < indexToTagData.size(); j++) {
            ArrayList<Double> jEmissions = new ArrayList<>();
            double denom = denoms.get(j);
            for(int k = 0; k < indexToWordData.size(); k++) {
                double num = 1;
                if(wordStateCount.containsKey(new Pair(j,k))) {
                    num += (wordStateCount.get(new Pair(j, k)));
                }
                jEmissions.add(num/denom);
            }
            toReturn.add(j, jEmissions);
        }
        return toReturn;
    }

    private static ArrayList<ArrayList<Double>> computeTransitionProbs(ArrayList<ArrayList<Pair<Integer, Integer>>> inputData, ArrayList<String> indexToTagData) {
        ArrayList<ArrayList<Double>> toReturn = new ArrayList<>();
        HashMap<Pair<Integer, Integer>, Integer>  transitionStateCount = new HashMap<>();
        ArrayList<Double> denoms = new ArrayList<>();
        for(ArrayList<Pair<Integer, Integer>> sentence : inputData) {
            for(int position = 0; position < sentence.size()-1; position++) {
                int j = sentence.get(position).getValue();
                int k = sentence.get(position+1).getValue();
                transitionStateCount.merge(new Pair<>(j, k), 1, Integer::sum);
            }
        }

        for(int j = 0; j < indexToTagData.size(); j++) {
            double denom = 0;
            for(int k = 0; k < indexToTagData.size(); k++) {
                if(transitionStateCount.containsKey(new Pair(j,k))) {
                    denom += (transitionStateCount.get(new Pair(j, k)));
                }
                denom += 1;
            }
            denoms.add(denom);
        }

        for(int j = 0; j < indexToTagData.size(); j++) {
            ArrayList<Double> jTransitions = new ArrayList<>();
            double denom = denoms.get(j);
            for(int k = 0; k < indexToTagData.size(); k++) {
                double num = 1;
                if(transitionStateCount.containsKey(new Pair(j,k))) {
                    num += (transitionStateCount.get(new Pair(j, k)));
                }
                jTransitions.add(num/denom);
            }
            toReturn.add(j, jTransitions);
        }

        return toReturn;
    }

    private static ArrayList<Double> computeInitialProbs(ArrayList<ArrayList<Pair<Integer, Integer>>> inputData, ArrayList<String> indexToTagData) {
        ArrayList<Double> toReturn = new ArrayList<>();
        HashMap<Integer, Integer> firstStateCount = new HashMap<>();

        for(ArrayList<Pair<Integer, Integer>> sentence : inputData) {
            firstStateCount.merge(sentence.get(0).getValue(), 1, Integer::sum);
        }

        double denom = 0;
        for(int tag = 0; tag < indexToTagData.size(); tag++) {
            if(firstStateCount.containsKey(tag)) {
                denom += (firstStateCount.get(tag));
            }
            denom += 1;
        }
        for(int tag = 0; tag < indexToTagData.size(); tag++) {
            double num = 1;
            if(firstStateCount.containsKey(tag)) {
                num += (firstStateCount.get(tag));
            }
            toReturn.add(tag, (num/denom));
            System.out.println("For tag " + indexToTagData.get(tag) + ", the initial prob is " + (num/denom));
        }
        return toReturn;
    }

    private static ArrayList<String> readIndexData(String inputFileName) {
        ArrayList<String> toReturn = new ArrayList<>();

        BufferedReader reader;
        String input;
        try {
            reader = new BufferedReader(new FileReader(inputFileName));
            while((input = reader.readLine()) != null) {
                toReturn.add(input);
            }
        } catch (NullPointerException e) {
            System.err.println("Null pointer error: " + e.getMessage());
        } catch (IOException e) {
            System.err.println("IO Error: " + e.getMessage());
        }
        return toReturn;
    }

    private static ArrayList<ArrayList<Pair<Integer, Integer>>> readInput(String inputFileName, ArrayList<String> indexToWordData, ArrayList<String> indexToTagData) {
        ArrayList<ArrayList<Pair<Integer, Integer>>> toReturn = new ArrayList<>();

        BufferedReader reader;
        String input;
        try {
            reader = new BufferedReader(new FileReader(inputFileName));
            while((input = reader.readLine()) != null ) {
                ArrayList<Pair<Integer, Integer>> line = new ArrayList<>();
                String[] pairs = input.split("\\s");
                for(int i = 0; i < pairs.length; i++) {
                    String pair = pairs[i];
                    pair = pair.replaceAll("<", "");
                    pair = pair.replaceAll(">", "");
                    String[] wordTagPair = pair.split("_");
                    int word = indexToWordData.indexOf(wordTagPair[0]);
                    int tag = indexToTagData.indexOf(wordTagPair[1]);
                    line.add(new Pair(word, tag));
                }
                toReturn.add(line);
            }
        } catch (NullPointerException e) {
            System.err.println("Null pointer error: " + e.getMessage());
        } catch (IOException e) {
            System.err.println("IO Error: " + e.getMessage());
        }
        return toReturn;
    }

    private static void printInitialProbs(String outputFileName, ArrayList<Double> initialProbs) {
        BufferedWriter writer;
        try {
            writer = new BufferedWriter(new FileWriter(outputFileName));
            String s;
            StringBuilder sb = new StringBuilder();
            for(double prob : initialProbs) {
                sb.append(prob);
                sb.append('\n');
            }
            s = sb.toString();
            writer.write(s);
            writer.close();
            System.out.println("Finished writing initial probabilities file.");
        } catch (Exception e) {
            System.err.println(e.getMessage());
        }
    }
    private static void printTransitionOrEmission(String outputFileName, ArrayList<ArrayList<Double>> probs) {
        BufferedWriter writer;
        try {
            writer = new BufferedWriter(new FileWriter(outputFileName));
            String s;
            StringBuilder sb = new StringBuilder();
            for(ArrayList<Double> row : probs) {
                for(double prob : row) {
                    sb.append(prob);
                    sb.append(" ");
                }
                sb.append('\n');
            }
            s = sb.toString();
            writer.write(s);
            writer.close();
            if(outputFileName.contains("em")) {
                System.out.println("Finished writing emission probabilities file.");
            } else if (outputFileName.contains("tr")) {
                System.out.println("Finished writing transition probabilities file.");
            } else {
                System.out.println("Finished writing probabilities file.");
            }
        } catch (Exception e) {
            System.err.println(e.getMessage());
        }
    }
    public static class Pair<F, S> extends java.util.AbstractMap.SimpleImmutableEntry<F, S> {
        public Pair(F f, S s) {
            super(f, s);
        }
    }
}