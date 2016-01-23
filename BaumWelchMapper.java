/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.classifier.sequencelearning.hmm.hadoop;

import java.io.IOException;
import java.util.HashMap;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.classifier.sequencelearning.hmm.HmmAlgorithms;
import org.apache.mahout.classifier.sequencelearning.hmm.HmmModel;
import org.apache.mahout.classifier.sequencelearning.hmm.HmmUtils;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * The mappers perform the first part of the Expectation step by calculating expected counts using the current iteration's model parameters.
 * The reducers finish the Expectation started here,
 * and also perform the Maximization step to calculate the parameters for the next MapReduce iteration.
 * <p/>
 * The input consists of sequence file with LongWritable keys denoting sequence IDs and the Values as Vector Writable wrapping the observed integer sequence.
 * The goal of the Mapper is to calculate and emit posterior probabilities for its input split.
 * The probabilities are wrapped in MapWritables.
 * <p/>
 * Each mapper generates 2n + 1 unique keys, where n is the number of hidden states, with the following scheme:
 * <p/>
 * For Initial Distribution:
 * key = INITIAL
 * value = MapWritable<hiddenStateID, initial probability>
 * <p/>
 * For Transition Distribution:
 * key = TRANSIT_0, TRANSIT_1 etc.
 * value =  MapWritable<hiddenStateID, transition probability>
 * <p/>
 * The suffixes to TRANSIT_ denote the hidden state from which the transitions are encoded in the value map
 * <p/>
 * eg: key = TRANSIT_0, value = <(0, 0.1), (1, 0.8), (2, 0.1)>
 * <p/>
 * For Emission Distribution:
 * key = EMIT_0, EMIT_1 etc.
 * value =  MapWritable<emissionStateID, emission probability>
 * <p/>
 * The suffixes to EMIT_ denote the hidden state from which the emissions are encoded in the value map
 * <p/>
 * eg: key = EMIT_0, value = <(0, 0.1), (1, 0.8), (2, 0.1)>
 */


public class BaumWelchMapper extends
  Mapper<LongWritable, VectorWritable, Text, MapWritable> {

  private static final Logger log = LoggerFactory.getLogger(BaumWelchMapper.class);

  private Integer nrOfHiddenStates;
  private Integer nrOfEmittedStates;
  private Path modelPath;
  private HmmModel Model;
  private HmmAlgorithms.ScalingMethod scaling = HmmAlgorithms.ScalingMethod.NOSCALING;

  @Override
  public void setup(Context context) throws IOException, InterruptedException {

    super.setup(context);
    Configuration config = context.getConfiguration();

    String scalingMethod = config.get(BaumWelchConfigKeys.SCALING_OPTION_KEY);

	if (scalingMethod.equals("rescaling")) {
		scaling = HmmAlgorithms.ScalingMethod.RESCALING;
	}
	else 	if (scalingMethod.equals("logscaling")) {
		scaling = HmmAlgorithms.ScalingMethod.LOGSCALING;
	}
	
    nrOfHiddenStates = Integer.parseInt(config.get(BaumWelchConfigKeys.NUMBER_OF_HIDDEN_STATES_KEY));
    nrOfEmittedStates = Integer.parseInt(config.get(BaumWelchConfigKeys.NUMBER_OF_EMITTED_STATES_KEY));
    MapWritable hiddenStatesWritableMap = MapWritableCache.load(config, new Path(config.get(BaumWelchConfigKeys.HIDDEN_STATES_MAP_PATH)));
    MapWritable emittedStatesWritableMap = MapWritableCache.load(config, new Path(config.get(BaumWelchConfigKeys.EMITTED_STATES_MAP_PATH)));

    String[] hiddenStatesArray = new String[hiddenStatesWritableMap.size()];
    String[] emittedStatesArray = new String[emittedStatesWritableMap.size()];

    int k = 0;
    int l = 0;

    for (MapWritable.Entry<Writable, Writable> entry : hiddenStatesWritableMap.entrySet()) {
      hiddenStatesArray[k++] = ((entry.getKey())).toString();
    }

    for (MapWritable.Entry<Writable, Writable> entry : emittedStatesWritableMap.entrySet()) {
      emittedStatesArray[l++] = ((entry.getKey())).toString();
    }

    modelPath = new Path(config.get(BaumWelchConfigKeys.MODEL_PATH_KEY));
    Model = BaumWelchUtils.createHmmModel(nrOfHiddenStates, nrOfEmittedStates, modelPath, config);
    Model.registerHiddenStateNames(hiddenStatesArray);
    Model.registerOutputStateNames(emittedStatesArray);
    HmmUtils.normalizeModel(Model);
    HmmUtils.validate(Model);

    log.info("Mapper Setup Hmm Model Created. Hidden States = {} Emitted States = {}", Model.getNrOfHiddenStates(), Model.getNrOfOutputStates());

  }

  @Override
  public void map(LongWritable seqID, VectorWritable seq, Context context)
    throws IOException, InterruptedException {

    MapWritable initialDistributionStripe = new MapWritable();
    HashMap<Integer, MapWritable> transitionDistributionStripe = new HashMap<Integer, MapWritable>();
    HashMap<Integer, MapWritable> emissionDistributionStripe = new HashMap<Integer, MapWritable>();

    Vector vec = seq.get();
    log.info("Sequence Length = {}", vec.size());
    int[] sequence = new int[vec.size()];

    int n = 0;

    for (int idx = 0; idx < vec.size(); idx++) {
	int val = (int) (vec.getElement(idx)).get();
      sequence[n] = val;
      n++;
    }

    if (scaling == HmmAlgorithms.ScalingMethod.LOGSCALING) {
		Matrix alphaFactors = HmmAlgorithms.forwardAlgorithm(Model, sequence, HmmAlgorithms.ScalingMethod.LOGSCALING, null);
      Matrix betaFactors = HmmAlgorithms.backwardAlgorithm(Model, sequence, HmmAlgorithms.ScalingMethod.LOGSCALING, null);

      //Initial Distribution
      for (int q = 0; q < nrOfHiddenStates; q++) {
        double alpha_1_q = alphaFactors.get(0, q);
        double beta_1_q = betaFactors.get(0, q);
        if ((alpha_1_q + beta_1_q) > Double.NEGATIVE_INFINITY) {
          initialDistributionStripe.put(new IntWritable(q), new DoubleWritable(alpha_1_q + beta_1_q));
        }
      }


      //Transition Distribution
      double[][] transitionMatrix = new double[nrOfHiddenStates][nrOfHiddenStates];
      for (int q = 0; q < nrOfHiddenStates; q++) {
        for (int x = 0; x < nrOfHiddenStates; x++) {
          transitionMatrix[q][x] = Double.NEGATIVE_INFINITY;
        }
      }

      for (int t = 0; t < sequence.length - 1; t++) {
        for (int q = 0; q < nrOfHiddenStates; q++) {
          for (int r = 0; r < nrOfHiddenStates; r++) {
            double alpha_t_q = alphaFactors.get(t, q);
            double A_q_r = Model.getTransitionMatrix().get(q, r) > 0 ?
              Math.log(Model.getTransitionMatrix().get(q, r)) : Double.NEGATIVE_INFINITY;
            double B_r_xtplus1 = Model.getEmissionMatrix().get(r, sequence[t + 1]) > 0 ?
              Math.log(Model.getEmissionMatrix().get(r, sequence[t + 1])) : Double.NEGATIVE_INFINITY;
            double beta_tplus1_r = betaFactors.get(t + 1, r);
            double transitionProb = alpha_t_q + A_q_r + B_r_xtplus1 + beta_tplus1_r;
            if (transitionProb > Double.NEGATIVE_INFINITY) {
              transitionMatrix[q][r] = transitionProb + Math.log(1 + Math.exp(transitionMatrix[q][r] - transitionProb));
            }
          }
        }
      }
      for (int q = 0; q < nrOfHiddenStates; q++) {
        MapWritable innerMap = new MapWritable();
        for (int r = 0; r < nrOfHiddenStates; r++) {
          if (transitionMatrix[q][r] > Double.NEGATIVE_INFINITY) {
            innerMap.put(new IntWritable(r), new DoubleWritable(transitionMatrix[q][r]));
          }
        }
        transitionDistributionStripe.put(q, innerMap);
      }


      //Emission distribution
      double[][] emissionMatrix = new double[nrOfHiddenStates][nrOfEmittedStates];
      for (int q = 0; q < nrOfHiddenStates; q++) {
        for (int x = 0; x < nrOfEmittedStates; x++) {
          emissionMatrix[q][x] = Double.NEGATIVE_INFINITY;
        }
      }
      for (int t = 0; t < sequence.length; t++) {
        for (int q = 0; q < nrOfHiddenStates; q++) {
          double alpha_t_q = alphaFactors.get(t, q);
          double beta_t_q = betaFactors.get(t, q);
          double sum = alpha_t_q + beta_t_q;
          double max = sum > emissionMatrix[q][sequence[t]] ? sum : emissionMatrix[q][sequence[t]];
          if (sum > Double.NEGATIVE_INFINITY) {
            emissionMatrix[q][sequence[t]] = sum + Math.log(1 + Math.exp(emissionMatrix[q][sequence[t]] - sum));
          }

        }
      }
      for (int q = 0; q < nrOfHiddenStates; q++) {
        MapWritable innerMap = new MapWritable();
        for (int r = 0; r < nrOfEmittedStates; r++) {
          if (emissionMatrix[q][r] > Double.NEGATIVE_INFINITY) {
            innerMap.put(new IntWritable(r), new DoubleWritable(emissionMatrix[q][r]));
          }
        }
        emissionDistributionStripe.put(q, innerMap);
      }
    } else if (scaling == HmmAlgorithms.ScalingMethod.RESCALING) {
		double[] scalingFactors = new double[vec.size()];
		
		Matrix alphaFactors = HmmAlgorithms.forwardAlgorithm(Model, sequence, HmmAlgorithms.ScalingMethod.RESCALING, scalingFactors);
		Matrix betaFactors = HmmAlgorithms.backwardAlgorithm(Model, sequence, HmmAlgorithms.ScalingMethod.RESCALING, scalingFactors);

      //Initial Distribution
      for (int q = 0; q < nrOfHiddenStates; q++) {
        double alpha_1_q = alphaFactors.get(0, q);
        double beta_1_q = betaFactors.get(0, q);
	initialDistributionStripe.put(new IntWritable(q), new DoubleWritable(alpha_1_q * beta_1_q / scalingFactors[0]));
      }

	  //Transition Distribution
      double[][] transitionMatrixNum = new double[nrOfHiddenStates][nrOfHiddenStates];
      double[][] transitionMatrixDenom = new double[nrOfHiddenStates][nrOfHiddenStates];
      for (int q = 0; q < nrOfHiddenStates; q++) {
        for (int x = 0; x < nrOfHiddenStates; x++) {
	    transitionMatrixNum[q][x] = 0.0;
	    transitionMatrixDenom[q][x] = 0.0;
        }
      }

      for (int t = 0; t < sequence.length - 1; t++) {
        for (int q = 0; q < nrOfHiddenStates; q++) {
          for (int r = 0; r < nrOfHiddenStates; r++) {
            double alpha_t_q = alphaFactors.get(t, q);
            double A_q_r = Model.getTransitionMatrix().get(q, r);
            double B_r_xtplus1 = Model.getEmissionMatrix().get(r, sequence[t + 1]);
            double beta_tplus1_r = betaFactors.get(t + 1, r);
            double beta_t_q = betaFactors.get(t, q);
            double transitionProbNum = alpha_t_q * A_q_r * B_r_xtplus1 * beta_tplus1_r;
            double transitionProbDenom = alpha_t_q * beta_t_q/scalingFactors[t];
            transitionMatrixNum[q][r] += transitionProbNum;
            transitionMatrixDenom[q][r] += transitionProbDenom;
          }
        }
      }
      for (int q = 0; q < nrOfHiddenStates; q++) {
        MapWritable innerMap = new MapWritable();
        for (int r = 0; r < nrOfHiddenStates; r++) {
	    byte [] doublePair = BaumWelchUtils.doublePairToByteArray(transitionMatrixNum[q][r], transitionMatrixDenom[q][r]);
	    innerMap.put(new IntWritable(r), new BytesWritable(doublePair));
        }
        transitionDistributionStripe.put(q, innerMap);
      }


      //Emission distribution
      double[][] emissionMatrixNum = new double[nrOfHiddenStates][nrOfEmittedStates];
      double[][] emissionMatrixDenom = new double[nrOfHiddenStates][nrOfEmittedStates];
      for (int q = 0; q < nrOfHiddenStates; q++) {
        for (int x = 0; x < nrOfEmittedStates; x++) {
          emissionMatrixNum[q][x] = 0.0;
          emissionMatrixDenom[q][x] = 0.0;
        }
      }

      for (int q = 0; q < nrOfHiddenStates; ++q) {
	  for (int j = 0; j < nrOfEmittedStates; ++j) {
	      double temp = 0;
	      double temp1 = 0;
	      for (int t = 0; t < sequence.length; ++t) {
		  // delta tensor
		  if (sequence[t] == j) {
		      temp += alphaFactors.get(t, q) * betaFactors.get(t, q)/scalingFactors[t];
		  }
		  temp1 += alphaFactors.get(t, q) * betaFactors.get(t, q)/scalingFactors[t];
	      }
	      emissionMatrixNum[q][j] += temp;
	      emissionMatrixDenom[q][j] += temp1;
	  }
      }

      for (int q = 0; q < nrOfHiddenStates; q++) {
        MapWritable innerMap = new MapWritable();
        for (int r = 0; r < nrOfEmittedStates; r++) {
	    byte [] doublePair = BaumWelchUtils.doublePairToByteArray(emissionMatrixNum[q][r], emissionMatrixDenom[q][r]);
	    innerMap.put(new IntWritable(r), new BytesWritable(doublePair));
        }
        emissionDistributionStripe.put(q, innerMap);
      }
    } else {
		Matrix alphaFactors = HmmAlgorithms.forwardAlgorithm(Model, sequence, HmmAlgorithms.ScalingMethod.NOSCALING, null);
		Matrix betaFactors = HmmAlgorithms.backwardAlgorithm(Model, sequence, HmmAlgorithms.ScalingMethod.NOSCALING, null);


      //Initial Distribution
      for (int q = 0; q < nrOfHiddenStates; q++) {
        double alpha_1_q = alphaFactors.get(0, q);
        double beta_1_q = betaFactors.get(0, q);
        initialDistributionStripe.put(new IntWritable(q), new DoubleWritable(alpha_1_q * beta_1_q));
      }


      //Transition Distribution
      double[][] transitionMatrix = new double[nrOfHiddenStates][nrOfHiddenStates];
      for (int q = 0; q < nrOfHiddenStates; q++) {
        for (int x = 0; x < nrOfHiddenStates; x++) {
          transitionMatrix[q][x] = 0.0;
        }
      }

      for (int t = 0; t < sequence.length - 1; t++) {
        for (int q = 0; q < nrOfHiddenStates; q++) {
          for (int r = 0; r < nrOfHiddenStates; r++) {
            double alpha_t_q = alphaFactors.get(t, q);
            double A_q_r = Model.getTransitionMatrix().get(q, r);
            double B_r_xtplus1 = Model.getEmissionMatrix().get(r, sequence[t + 1]);
            double beta_tplus1_r = betaFactors.get(t + 1, r);
            double transitionProb = alpha_t_q * A_q_r * B_r_xtplus1 * beta_tplus1_r;
            transitionMatrix[q][r] += transitionProb;
          }
        }
      }
      for (int q = 0; q < nrOfHiddenStates; q++) {
        MapWritable innerMap = new MapWritable();
        for (int r = 0; r < nrOfHiddenStates; r++) {
          innerMap.put(new IntWritable(r), new DoubleWritable(transitionMatrix[q][r]));
        }
        transitionDistributionStripe.put(q, innerMap);
      }


      //Emission distribution
      double[][] emissionMatrix = new double[nrOfHiddenStates][nrOfEmittedStates];
      for (int q = 0; q < nrOfHiddenStates; q++) {
        for (int x = 0; x < nrOfEmittedStates; x++) {
          emissionMatrix[q][x] = 0.0;
        }
      }
      for (int t = 0; t < sequence.length; t++) {
        for (int q = 0; q < nrOfHiddenStates; q++) {

          double alpha_t_q = alphaFactors.get(t, q);
          double beta_t_q = betaFactors.get(t, q);
          emissionMatrix[q][sequence[t]] += alpha_t_q * beta_t_q;

        }
      }
      for (int q = 0; q < nrOfHiddenStates; q++) {
        MapWritable innerMap = new MapWritable();
        for (int r = 0; r < nrOfEmittedStates; r++) {
          innerMap.put(new IntWritable(r), new DoubleWritable(emissionMatrix[q][r]));
        }
        emissionDistributionStripe.put(q, innerMap);
      }

    }

    //push out the associative arrays
    context.write(new Text("INITIAL"), initialDistributionStripe);
    for (int q = 0; q < nrOfHiddenStates; q++) {
      context.write(new Text("EMIT_" + Integer.toString(q)), emissionDistributionStripe.get(q));
      context.write(new Text("TRANSIT_" + Integer.toString(q)), transitionDistributionStripe.get(q));
    }


  }
}
