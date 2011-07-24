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

package org.apache.mahout.classifier.sequencelearning.baumwelchmapreduce;

import java.io.IOException;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Mapper;

import org.apache.lucene.analysis.CharArrayMap;
import org.apache.mahout.classifier.sequencelearning.hmm.HmmAlgorithms;
import org.apache.mahout.classifier.sequencelearning.hmm.HmmModel;
import org.apache.mahout.classifier.sequencelearning.hmm.HmmUtils;
import org.apache.mahout.clustering.kmeans.KMeansConfigKeys;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class BaumWelchMapper extends
  Mapper<LongWritable, IntArrayWritable, Text, MapWritable> {

  private static final Logger log = LoggerFactory.getLogger(BaumWelchMapper.class);

  private Integer nrOfHiddenStates;
  private Integer nrOfEmittedStates;
  private Path modelPath;
  private HmmModel Model;

  @Override
  public void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    Configuration config = context.getConfiguration();

    nrOfHiddenStates = Integer.parseInt(config.get(BaumWelchConfigKeys.NUMBER_OF_HIDDEN_STATES_KEY));
    nrOfEmittedStates = Integer.parseInt(config.get(BaumWelchConfigKeys.NUMBER_OF_EMITTED_STATES_KEY));
    MapWritable hiddenStatesWritableMap = MapWritableCache.load(config, new Path(config.get(BaumWelchConfigKeys.HIDDEN_STATES_MAP_PATH)));
    log.info("Mapper Setup hiddenStatesWritableMap loaded. Number of entries = {}", hiddenStatesWritableMap.size());
    MapWritable emittedStatesWritableMap = MapWritableCache.load(config, new Path(config.get(BaumWelchConfigKeys.EMITTED_STATES_MAP_PATH)));
    log.info("Mapper Setup emittedStatesWritableMap loaded. Number of entries = {}", emittedStatesWritableMap.size());

    //HashMap hiddenStatesMap = new HashMap();
    //HashMap emittedStatesMap = new HashMap();

    String[] hiddenStatesArray = new String[hiddenStatesWritableMap.size()];
    String[] emittedStatesArray = new String[emittedStatesWritableMap.size()];

    int k = 0;
    int l = 0;

    for (MapWritable.Entry<Writable, Writable> entry : hiddenStatesWritableMap.entrySet()) {
      log.info("Mapper Setup hiddenStateMap adding pair ({} ,{})", ((Text) (entry.getKey())).toString(), ((IntWritable) (entry.getValue())).get());
      //hiddenStatesMap.put( ((Text)(entry.getKey())).toString(), ((IntWritable)(entry.getValue())).get() );
      hiddenStatesArray[k++] = ((Text) (entry.getKey())).toString();
    }

    for (MapWritable.Entry<Writable, Writable> entry : emittedStatesWritableMap.entrySet()) {
      log.info("Mapper Setup emittedStateMap adding pair ({} ,{})", ((Text) (entry.getKey())).toString(), ((IntWritable) (entry.getValue())).get());
      //emittedStatesMap.put( ((Text)(entry.getKey())).toString(), ((IntWritable)(entry.getValue())).get() );
      emittedStatesArray[l++] = ((Text) (entry.getKey())).toString();
    }


    modelPath = new Path(config.get(BaumWelchConfigKeys.MODEL_PATH_KEY));
    Model = BaumWelchUtils.CreateHmmModel(nrOfHiddenStates, nrOfEmittedStates, modelPath, config);
    Model.registerHiddenStateNames(hiddenStatesArray);
    Model.registerOutputStateNames(emittedStatesArray);
    HmmUtils.validate(Model);

    log.info("Mapper Setup Hmm Model Created. Hidden States = {} Emitted States = {}", Model.getNrOfHiddenStates(), Model.getNrOfOutputStates());
    Vector initialPr = Model.getInitialProbabilities();
    Matrix transitionPr = Model.getTransitionMatrix();
    Matrix emissionPr = Model.getEmissionMatrix();


    for (int i = 0; i < Model.getNrOfHiddenStates(); i++) {
      log.info("Mapper Setup Hmm Model Initial Prob Vector. State {} = {}", i, initialPr.get(i));
    }


    for (int i = 0; i < Model.getNrOfHiddenStates(); i++) {
      for (int j = 0; j < Model.getNrOfHiddenStates(); j++) {
        log.info("Mapper Setup Hmm Model Transition Prob Matrix ({}, {}) = {} ", new Object[]{i, j, transitionPr.get(i, j)});
      }
    }


    for (int i = 0; i < Model.getNrOfHiddenStates(); i++) {
      for (int j = 0; j < Model.getNrOfOutputStates(); j++) {
        log.info("Mapper Setup Hmm Model Emission Prob Matrix. ({}, {}) = {}", new Object[]{i, j, emissionPr.get(i, j)});
      }
    }
  }

  @Override
  public void map(LongWritable seqID, IntArrayWritable seq, Context context)
    throws IOException, InterruptedException {

    MapWritable initialDistributionStripe = new MapWritable();
    MapWritable transitionDistributionStripe = new MapWritable();
    MapWritable emissionDistributionStripe = new MapWritable();


    //IntArrayWritable[] writableSequence = (IntArrayWritable[])seq.get();
    //int[] sequence = new int[seq.get().length];
    int[] sequence = new int[seq.get().length];

    int n = 0;
    for (Writable val : seq.get()) {
      sequence[n] = ((IntWritable) val).get();
      n++;
    }

    for (int k = 0; k < sequence.length; k++) {
      log.info("Sequence Array {}", Integer.toString(sequence[k]));
    }


    Matrix alphaFactors = HmmAlgorithms.forwardAlgorithm(Model, sequence, false);
    for (int i = 0; i < alphaFactors.numRows(); i++) {
      for (int j = 0; j < alphaFactors.numCols(); j++) {
        log.info("Alpha Factors Matrix entry ({}, {}) = {}", new Object[]{i, j, alphaFactors.get(i, j)});
      }
    }


    Matrix betaFactors = HmmAlgorithms.backwardAlgorithm(Model, sequence, false);
    for (int i = 0; i < betaFactors.numRows(); i++) {
      for (int j = 0; j < betaFactors.numCols(); j++) {
        log.info("Beta Factors Matrix entry ({}, {}) = {}", new Object[]{i, j, betaFactors.get(i, j)});
      }

      //Initial Distribution
      for (int q = 0; q < nrOfHiddenStates; q++) {
        double alpha_1_q = alphaFactors.get(1, q);
        double beta_1_q = betaFactors.get(1, q);
        initialDistributionStripe.put(new IntWritable(q), new DoubleWritable(alpha_1_q * beta_1_q));
      }

      //Emission Distribution
      /*
    Matrix emissionMatrix = new DenseMatrix(nrOfHiddenStates, sequence.length);
    for (int t = 0; t < sequence.length; t++) {
      HashMap<Integer, Double> innerMap = new HashMap<Integer, Double>();
      for (int q = 0; q < nrOfHiddenStates; q++) {
        double alpha_t_q = alphaFactors.get(t, q);
        double beta_t_q  = betaFactors.get(t, q);
        //innerMap.put(q, alpha_t_q * beta_t_q);
        emissionMatrix.set(q, t, alpha_t_q * beta_t_q);
        }
    }
    for (int q = 0; q < nrOfHiddenStates; q++) {
      Map innerEmissionMap = new MapWritable();
      for (int xt = 0; xt < sequence.length; xt++) {
        innerEmissionMap.put(new IntWritable(xt), new DoubleWritable(emissionMatrix.get(q, xt)));
      }
      emissionDistributionStripe.put(new IntWritable(q), (MapWritable)innerEmissionMap);
    }
    */


      double[][] emissionMatrix = new double[nrOfHiddenStates][nrOfEmittedStates];

      for (int q = 0; q < nrOfHiddenStates; q++) {
        for (int x = 0; x < nrOfEmittedStates; x++) {
          emissionMatrix[q][x] = 0.0;
        }
      }

      for (int t = 0; t < sequence.length; t++) {
        //HashMap<Integer, Double> innerMap = new HashMap<Integer, Double>();
        for (int q = 0; q < nrOfHiddenStates; q++) {
          double alpha_t_q = alphaFactors.get(t, q);
          double beta_t_q = betaFactors.get(t, q);
          //innerMap.put(q, alpha_t_q * beta_t_q);
          //emissionMatrix.set(q, t, alpha_t_q * beta_t_q);
          emissionMatrix[q][sequence[t]] += alpha_t_q * beta_t_q;
        }
      }
      for (int q = 0; q < nrOfHiddenStates; q++) {
        Map innerEmissionMap = new MapWritable();
        for (int xt = 0; xt < sequence.length; xt++) {
          innerEmissionMap.put(new IntWritable(sequence[xt]), new DoubleWritable(emissionMatrix[q][sequence[xt]]));
        }
        emissionDistributionStripe.put(new IntWritable(q), (MapWritable) innerEmissionMap);
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
            log.info("Putting into Inner Map of Transition Distribution. Key = {}, Value = {}", q, transitionProb);
            transitionMatrix[q][r] += transitionProb;
          }
        }
      }
      for (int q = 0; q < nrOfHiddenStates; q++) {
        Map innerTransitionMap = new MapWritable();
        for (int r = 0; r < nrOfHiddenStates; r++) {
          innerTransitionMap.put(new IntWritable(r), new DoubleWritable(transitionMatrix[q][r]));
        }
        transitionDistributionStripe.put(new IntWritable(q), (MapWritable) innerTransitionMap);
      }


      context.write(new Text("INITIAL"), initialDistributionStripe);
      log.info("Context Writing from Mapper the Initial Distribution Stripe. Size = {}  Entries = {}", Integer.toString(initialDistributionStripe.size()), Integer.toString(initialDistributionStripe.entrySet().size()));
      for (int q = 0; q < nrOfHiddenStates; q++) {
        context.write(new Text("EMIT_" + Integer.toString(q)), (MapWritable) emissionDistributionStripe.get(new IntWritable(q)));
        log.info("Context Writing from Mapper the Emission Distribution Stripe. State = {}  Entries = {}", Integer.toString(q), Integer.toString(((MapWritable) emissionDistributionStripe.get(new IntWritable(q))).size()));
        for (MapWritable.Entry<Writable, Writable> entry : ((MapWritable) emissionDistributionStripe.get(new IntWritable(q))).entrySet()) {
          log.info("Emission Distribution Stripe Details. Key = {}  Value = {} ", Integer.toString(((IntWritable) entry.getKey()).get()), Double.toString(((DoubleWritable) entry.getValue()).get()));
        }
        context.write(new Text("TRANSIT_" + Integer.toString(q)), (MapWritable) transitionDistributionStripe.get(new IntWritable(q)));
        log.info("Context Writing from Mapper the Transition Distribution Stripe. State = {}  Entries = {}", Integer.toString(q), Integer.toString(((MapWritable) transitionDistributionStripe.get(new IntWritable(q))).size()));
        for (MapWritable.Entry<Writable, Writable> entry : ((MapWritable) transitionDistributionStripe.get(new IntWritable(q))).entrySet()) {
          log.info("Transition Distribution Stripe Details. Key = {}  Value = {} ", Integer.toString(((IntWritable) entry.getKey()).get()), Double.toString(((DoubleWritable) entry.getValue()).get()));
        }
      }

    }
  }


}
