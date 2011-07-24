/* Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
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
import java.io.DataInput;
import java.io.DataOutput;

import java.util.Map;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.MapWritable;

import org.apache.mahout.math.*;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.classifier.sequencelearning.hmm.HmmModel;


public class BaumWelchModel extends HmmModel implements Writable {

  /**
   * Bi-directional Map for storing the observed state names
   */
  private MapWritable outputStateNames;

  /**
   * Bi-Directional Map for storing the hidden state names
   */
  private MapWritable hiddenStateNames;

  /* Number of hidden states */
  private IntWritable nrOfHiddenStates;

  /**
   * Number of output states
   */
  private IntWritable nrOfOutputStates;

  /**
   * Transition matrix containing the transition probabilities between hidden
   * states. TransitionMatrix(i,j) is the probability that we change from hidden
   * state i to hidden state j In general: P(h(t+1)=h_j | h(t) = h_i) =
   * transitionMatrix(i,j) Since we have to make sure that each hidden state can
   * be "left", the following normalization condition has to hold:
   * sum(transitionMatrix(i,j),j=1..hiddenStates) = 1
   */
  private MatrixWritable transitionMatrix;

  /**
   * Output matrix containing the probabilities that we observe a given output
   * state given a hidden state. outputMatrix(i,j) is the probability that we
   * observe output state j if we are in hidden state i Formally: P(o(t)=o_j |
   * h(t)=h_i) = outputMatrix(i,j) Since we always have an observation for each
   * hidden state, the following normalization condition has to hold:
   * sum(outputMatrix(i,j),j=1..outputStates) = 1
   */
  private MatrixWritable emissionMatrix;

  /**
   * Vector containing the initial hidden state probabilities. That is
   * P(h(0)=h_i) = initialProbabilities(i). Since we are dealing with
   * probabilities the following normalization condition has to hold:
   * sum(initialProbabilities(i),i=1..hiddenStates) = 1
   */
  private VectorWritable initialProbabilities;


  public BaumWelchModel(HmmModel model) {

    this.transitionMatrix.set(model.getTransitionMatrix());
    this.emissionMatrix.set(model.getEmissionMatrix());
    this.initialProbabilities.set(model.getInitialProbabilities());
    this.nrOfHiddenStates.set(model.getNrOfHiddenStates());
    this.nrOfOutputStates.set(model.getNrOfOutputStates());

    Map<String, Integer> hiddenStates = model.getHiddenStateNames();
    Map<String, Integer> emitStates = model.getOutputStateNames();


    for (Map.Entry<String, Integer> e : hiddenStates.entrySet()) {
      this.hiddenStateNames.put(new Text(e.getKey()), new IntWritable(e.getValue()));
    }

    for (Map.Entry<String, Integer> e : emitStates.entrySet()) {
      this.outputStateNames.put(new Text(e.getKey()), new IntWritable(e.getValue()));
    }

  }


  @Override
  public void readFields(DataInput in) throws IOException {

    nrOfHiddenStates.readFields(in);
    nrOfOutputStates.readFields(in);
    hiddenStateNames.readFields(in);
    outputStateNames.readFields(in);
    transitionMatrix.readFields(in);
    emissionMatrix.readFields(in);
    initialProbabilities.readFields(in);

  }

  @Override
  public void write(DataOutput out) throws IOException {
    // serialize the model parameters
    nrOfHiddenStates.write(out);
    nrOfOutputStates.write(out);
    hiddenStateNames.write(out);
    outputStateNames.write(out);

    //serialize the probability matrices
    transitionMatrix.write(out);
    emissionMatrix.write(out);
    initialProbabilities.write(out);
  }
}

















