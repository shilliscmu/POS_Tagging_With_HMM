import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

public class forwardbackward {

    public static void main(String[] args) {
        String testInput = args[0];
        String indexToWord = args[1];
        String indexToTag = args[2];
        String hmmPrior = args[3];
        String hmmEmit = args[4];
        String hmmTrans = args[5];
        String predictedFile = args[6];
        String metricsFile = args[7];

        ArrayList<String> indexToWordData = readIndexData(indexToWord);
        ArrayList<String> indexToTagData = readIndexData(indexToTag);
        ArrayList<ArrayList<Pair<Integer, Integer>>> inputData = readInput(testInput, indexToWordData, indexToTagData);

        ArrayList<Double> initialProbsPi = readInitialProbs(hmmPrior);
        ArrayList<ArrayList<Double>> transitionProbsA = readTransitionOrEmissionProbs(hmmTrans);
        ArrayList<ArrayList<Double>> emissionProbsB = readTransitionOrEmissionProbs(hmmEmit);

//        ArrayList<ArrayList<String>> yhats = new ArrayList<>();
        StringBuilder sb = new StringBuilder();
        double numberOfTags = 0;
        double numberWrong = 0;
        double logLikelihood = 0;
        for(int i = 0; i < inputData.size(); i++) {
            ArrayList<Pair<Integer, Integer>> testWordsAndOldTags = inputData.get(i);

//            System.out.print("\nSentence: ");
//            for(Pair<Integer, Integer> wordOldTag : testWordsAndOldTags) {
//                System.out.print(indexToWordData.get(wordOldTag.getKey()) + " ");
//            }

            ArrayList<ArrayList<Double>> alpha = forward(initialProbsPi, transitionProbsA, emissionProbsB, inputData, indexToTagData, i);

//            System.out.println("Alpha: ");
//            for(int row = 0; row < alpha.size(); row++) {
//                System.out.println(alpha.get(row).toString());
//            }
//            System.out.println();

            ArrayList<ArrayList<Double>> beta = backward(transitionProbsA, emissionProbsB, inputData, i);

//            System.out.println("Beta: ");
//            for(int row = 0; row < beta.size(); row++) {
//                System.out.println(beta.get(row).toString());
//            }
//            System.out.println();

            ArrayList<String> yhats = minBayesRiskPrediction(alpha, beta, indexToTagData);
            for(int position = 0; position < testWordsAndOldTags.size(); position++) {
                Pair<Integer, Integer> wordAndOldTag = testWordsAndOldTags.get(position);

                if(position != 0) {
                    sb.append(" ");
                }

                sb.append(indexToWordData.get(wordAndOldTag.getKey()));
                sb.append("_");
                sb.append(yhats.get(position));

                numberOfTags++;
                if(!yhats.get(position).equals(indexToTagData.get(wordAndOldTag.getValue()))) {
                    numberWrong++;
                }
            }
            sb.append('\n');

            double sum = 0;
            for(double prob : alpha.get(alpha.size()-1)) {
                sum += prob;
            }
            logLikelihood += Math.log(sum);
        }
        printYhats(predictedFile, sb);

        logLikelihood /= inputData.size();

        double accuracy = (numberOfTags - numberWrong) / numberOfTags;

        printMetrics(metricsFile, logLikelihood, accuracy);

//        System.out.println("Log Likelihood: " + logLikelihood);
//        System.out.println("Accuracy: " + accuracy);
    }
    private static void printMetrics(String outputFileName, double logLikelihood, double accuracy) {
        BufferedWriter writer;
        try {
            writer = new BufferedWriter(new FileWriter(outputFileName));
            StringBuilder sb = new StringBuilder();
            sb.append("Average Log-Likelihood: ");
            sb.append(logLikelihood);
            sb.append('\n');
            sb.append("Accuracy: ");
            sb.append(accuracy);
            writer.write(sb.toString());
            writer.close();
            System.out.println("Finished writing metrics file.");
        } catch (Exception e) {
            System.err.println(e.getMessage());
        }
    }

    private static void printYhats(String outputFileName, StringBuilder sb) {
        BufferedWriter writer;
        try {
            writer = new BufferedWriter(new FileWriter(outputFileName));
            writer.write(sb.toString());
            writer.close();
            System.out.println("Finished writing predicted file.");
        } catch (Exception e) {
            System.err.println(e.getMessage());
        }
    }
    private static ArrayList<String> minBayesRiskPrediction(ArrayList<ArrayList<Double>> alpha, ArrayList<ArrayList<Double>> beta, ArrayList<String> indexToTagData) {
        ArrayList<String> yhats = new ArrayList<>();

        int T = alpha.size();
        int J = alpha.get(0).size();
        for(int t = 0; t < T; t++) {
//            System.out.println("For word " + t + ": ");
            ArrayList<Double> alpha_t = alpha.get(t);
            ArrayList<Double> beta_t = beta.get(t);
            ArrayList<Double> product = new ArrayList<>();
            for(int j = 0; j < J; j++) {
                product.add(j, alpha_t.get(j)*beta_t.get(j));
//                System.out.println("The probability of state " + j + " is " + alpha_t.get(j)*beta_t.get(j));
            }
            double best = Double.NEGATIVE_INFINITY;
            int argmax = -1;
            for(int i = 0; i < product.size(); i++) {
                double candidate = product.get(i);
                if(candidate > best) {
                    best = candidate;
                    argmax = i;
                }
            }
            yhats.add(t, indexToTagData.get(argmax));
//            System.out.println("For word " + t + ", we picked state " + argmax + " (aka, " + "state " + indexToTagData.get(argmax) + ")");
        }

        return yhats;
    }
    private static ArrayList<ArrayList<Double>> backward(ArrayList<ArrayList<Double>> transitionProbs, ArrayList<ArrayList<Double>> emissionProbs, ArrayList<ArrayList<Pair<Integer, Integer>>> inputData, int index) {
        //[T = sequence length][J = number of tags]
        //normalize along rows, except last row
        int T = inputData.get(index).size();
        int J = emissionProbs.size();
        ArrayList<ArrayList<Double>> beta = new ArrayList<>();
        ArrayList<Double> betaT = new ArrayList<>();
        for(int j = 0; j < J; j++) {
            betaT.add(1.0);
        }
        for(int t = 0; t < T; t++) {
            beta.add(betaT);
        }

        double denom = 0;
        for(int j = 0; j < betaT.size(); j++) {
            denom += betaT.get(j);
        }
        for(int j = 0; j < betaT.size(); j++) {
            double num = betaT.get(j);
            betaT.set(j, num/denom);
        }

        beta.set(T-1, betaT);

        for(int t = T-2; t >= 0; t--) {
//            System.out.println("\nFor beta " + t + ": ");
            ArrayList<Double> beta_t = new ArrayList<>();
            ArrayList<Double> beta_tPlus1 = beta.get(t+1);
            for(int j = 0; j < J; j++) {
                double sum = 0;
//                System.out.println("\nFor state j=" + j + ": ");
                for(int k = 0; k < beta_tPlus1.size(); k++) {
                    double emissionsProb = emissionProbs.get(k).get(inputData.get(index).get(t+1).getKey());
                    double beta_tPlus1OfK = beta_tPlus1.get(k);
                    double transitionProb = transitionProbs.get(j).get(k);
//                    System.out.println("\nFor state k=" + k + ": ");
//                    System.out.println("At state " + k + ", the probability of emitting word " + (t+1) + " is " + emissionsProb);
//                    System.out.println("The probability of state " + k + " at beta " + (t+1) + " is " + beta_tPlus1OfK);
//                    System.out.println("At state " + j + ", the probability of transitioning to state " + k + " is " + transitionProb);
                    sum += (emissionsProb * beta_tPlus1OfK * transitionProb);
                }
                beta_t.add(sum);
            }

//            System.out.println("Pre-normalization Beta " + t + ": " + beta_t.toString());
            denom = 0;
            for(int j = 0; j < beta_t.size(); j++) {
                denom += beta_t.get(j);
            }
            for(int j = 0; j < beta_t.size(); j++) {
                double num = beta_t.get(j);
                beta_t.set(j, num/denom);
            }
            beta.set(t, beta_t);
        }
        return beta;
    }
    //B[state j][word k]
    private static ArrayList<ArrayList<Double>> forward(ArrayList<Double> initialProbs, ArrayList<ArrayList<Double>> transitionProbs, ArrayList<ArrayList<Double>> emissionProbs, ArrayList<ArrayList<Pair<Integer, Integer>>> inputData, ArrayList<String> indexToTagData, int index) {
        //[T = sequence length][J = number of tags]
        //normalize along rows, except last row
        ArrayList<ArrayList<Double>> alpha = new ArrayList<>();
        ArrayList<Double> alphaZero = new ArrayList<>();
//        System.out.println("For alpha 0:");
        for(int j = 0; j < initialProbs.size(); j++) {
            double initialProb_j = initialProbs.get(j);
            double emissionProb_j0 = emissionProbs.get(j).get(inputData.get(index).get(0).getKey());
//            System.out.println("At state " + j + ", the initial probability of j is " + initialProb_j);
//            System.out.println("At state " + j + ", the probability of emitting x0 from j is " + emissionProb_j0);
            alphaZero.add(initialProb_j * emissionProb_j0);
        }

//        System.out.println("Pre-normalization AlphaZero: " + alphaZero.toString() + '\n');

        double denom = 0;
        for(int j = 0; j < alphaZero.size(); j++) {
            denom += alphaZero.get(j);
        }
        for(int j = 0; j < alphaZero.size(); j++) {
            double num = alphaZero.get(j);
            alphaZero.set(j, num/denom);
        }
        alpha.add(alphaZero);

        int J = indexToTagData.size();
        int T = inputData.get(index).size();
        for(int t = 1; t < T; t++) {
//            System.out.println("\nFor alpha " + t + ": ");
            ArrayList<Double> alpha_t = new ArrayList<>();
            for(int j = 0; j < J; j++) {
                double emissionProb_jt = emissionProbs.get(j).get(inputData.get(index).get(t).getKey());
//                System.out.println("At state " + j + ", the probability of emitting word " + inputData.get(index).get(t).getKey() + " from j is " + emissionProb_jt);
                double sumOf_PastAlphaForEachTagTimesTransmissionProbFromThatTagTo_j = 0;
                ArrayList<Double> alpha_tMinus1 = alpha.get(t-1);
                for(int k = 0; k < alpha_tMinus1.size(); k++) {
//                    System.out.println("The probability for state " + k + " at alpha " + (t-1) + " is " + alpha_tMinus1.get(k));
//                    System.out.println("The probability of transitioning from state " + k + " to state " + j + " is " + transitionProbs.get(k).get(j));
                    double product = alpha_tMinus1.get(k) * transitionProbs.get(k).get(j);
                    sumOf_PastAlphaForEachTagTimesTransmissionProbFromThatTagTo_j += product;
                }
                alpha_t.add((emissionProb_jt * sumOf_PastAlphaForEachTagTimesTransmissionProbFromThatTagTo_j));
            }

//            System.out.println("Pre-normalization Alpha " + t + ": " + alpha_t.toString());

            if(t != (T-1)) {
                denom = 0;
                for(int j = 0; j < alpha_t.size(); j++) {
                    denom += alpha_t.get(j);
                }
                for(int j = 0; j < alpha_t.size(); j++) {
                    double num = alpha_t.get(j);
                    alpha_t.set(j, num/denom);
                }
            } else {
//                System.out.println("Alpha_T: " + alpha_t + "\n");
            }
            alpha.add(alpha_t);
//            System.out.println();
        }
        return alpha;
    }
    private static ArrayList<ArrayList<Double>> readTransitionOrEmissionProbs(String inputFileName) {
        ArrayList<ArrayList<Double>> toReturn = new ArrayList<>();
        BufferedReader reader;
        String input;
        try {
            reader = new BufferedReader(new FileReader(inputFileName));
            while((input = reader.readLine()) != null ) {
                ArrayList<Double> rowProbs = new ArrayList<>();
                String[] row = input.split(" ");
                for(int k = 0; k < row.length; k++) {
                    rowProbs.add(Double.parseDouble(row[k]));
                }
                toReturn.add(rowProbs);
            }
        } catch (NullPointerException e) {
            System.err.println("Null pointer error: " + e.getMessage());
        } catch (IOException e) {
            System.err.println("IO Error: " + e.getMessage());
        }
        return toReturn;
    }
    private static ArrayList<Double> readInitialProbs(String inputFileName) {
        ArrayList<Double> toReturn = new ArrayList<>();
        BufferedReader reader;
        String input;
        try {
            reader = new BufferedReader(new FileReader(inputFileName));
            while((input = reader.readLine()) != null ) {
                toReturn.add(Double.parseDouble(input));
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
    public static class Pair<F, S> extends java.util.AbstractMap.SimpleImmutableEntry<F, S> {
        public Pair(F f, S s) {
            super(f, s);
        }
    }
}
