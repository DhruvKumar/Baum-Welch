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

import com.google.common.io.Closeables;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileUtil;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.mahout.classifier.sequencelearning.hmm.HmmModel;
import org.apache.mahout.classifier.sequencelearning.hmm.HmmUtils;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.nio.ByteBuffer;

/**
 * Utilities to convert between HmmModel and Sequence File representation.
 */

public class BaumWelchUtils {

  private static final Logger log = LoggerFactory.getLogger(BaumWelchUtils.class);

  private BaumWelchUtils() {

  }

  /**
   * Converts the sequence files present in a directory to a {@link HmmModel} model.
   *
   * @param nrOfHiddenStates Number of hidden states
   * @param nrOfOutputStates Number of output states
   * @param modelPath        Location of the sequence files containing the model's distributions
   * @param conf             Configuration object
   * @return HmmModel the encoded model
   * @throws IOException
   */
  public static HmmModel createHmmModel(int nrOfHiddenStates,
                                        int nrOfOutputStates,
                                        Path modelPath,
                                        Configuration conf) throws IOException {


    log.info("Entering Create Hmm Model. Model Path = {}", modelPath.toUri());
    Vector initialProbabilities = new DenseVector(nrOfHiddenStates);
    Matrix transitionMatrix = new DenseMatrix(nrOfHiddenStates, nrOfHiddenStates);
    Matrix emissionMatrix = new DenseMatrix(nrOfHiddenStates, nrOfOutputStates);

    // Get the path location where the seq files encoding model are stored
    Path modelFilesPath = new Path(modelPath, "*");

    Collection<Path> result = new ArrayList<Path>();

    // get all filtered file names in result list
    FileSystem fs = modelFilesPath.getFileSystem(conf);
    FileStatus[] matches = fs.listStatus(FileUtil.stat2Paths(fs.globStatus(modelFilesPath, PathFilters.partFilter())),
      PathFilters.partFilter());

    for (FileStatus match : matches) {
      result.add(fs.makeQualified(match.getPath()));
    }

    // iterate through the result path list
    for (Path path : result) {
      for (Pair<Writable, MapWritable> pair : new SequenceFileIterable<Writable, MapWritable>(path, true, conf)) {
        Text key = (Text) pair.getFirst();
        MapWritable valueMap = pair.getSecond();
        if (key.charAt(0) == (int) 'I') {
          // initial distribution stripe
          for (MapWritable.Entry<Writable, Writable> entry : valueMap.entrySet()) {
            initialProbabilities.set(((IntWritable) entry.getKey()).get(), ((DoubleWritable) entry.getValue()).get());
          }
        } else if (key.charAt(0) == (int) 'T') {
          // transition distribution stripe
          // key is of the form TRANSIT_0, TRANSIT_1 etc
          int stateID = Integer.parseInt(key.toString().split("_")[1]);
          for (MapWritable.Entry<Writable, Writable> entry : valueMap.entrySet()) {
            transitionMatrix.set(stateID, ((IntWritable) entry.getKey()).get(), ((DoubleWritable) entry.getValue()).get());
          }
        } else if (key.charAt(0) == (int) 'E') {
          // emission distribution stripe
          // key is of the form EMIT_0, EMIT_1 etc
          int stateID = Integer.parseInt(key.toString().split("_")[1]);
          for (MapWritable.Entry<Writable, Writable> entry : valueMap.entrySet()) {
            emissionMatrix.set(stateID, ((IntWritable) entry.getKey()).get(), ((DoubleWritable) entry.getValue()).get());
          }
        } else {
          throw new IllegalStateException("Error creating HmmModel from Sequence File Path");
        }
      }
    }

    HmmModel model = new HmmModel(transitionMatrix, emissionMatrix, initialProbabilities);

    if (model != null) {
      return model;
    } else throw new IOException("Error building model from output location");

  }

  /**
   * Builds a random {@link HmmModel} encoded as a Sequence File and writes it to the specified location.
   *
   * @param numHidden   Number of hidden states
   * @param numObserved Number of observed states
   * @param modelPath   Directory path for storing the created HmmModel
   * @param conf        Configuration object
   * @throws IOException
   */

  public static void buildRandomModel(int numHidden,
                                      int numObserved,
                                      Path modelPath,
                                      Configuration conf) throws IOException {
    HmmModel model = new HmmModel(numHidden, numObserved);
    HmmUtils.validate(model);
    writeModelToDirectory(model, modelPath, conf);
  }

  /**
   * Constructs a HmmModel object using the distributions and stores it as a sequence file.
   *
   * @param initialProb    initial hidden state probability distribution vector
   * @param transitionProb hidden state transition probability distribution matrix
   * @param emissionProb   emission probability distribution matrix
   * @param modelPath      path to store the constructed {@link HmmModel}
   * @param conf           Configuration object
   * @throws IOException
   */
  public static void buildHmmModelFromDistributions(double[] initialProb,
                                                    double[][] transitionProb,
                                                    double[][] emissionProb,
                                                    Path modelPath,
                                                    Configuration conf) throws IOException {
    HmmModel model = new HmmModel(new DenseMatrix(transitionProb),
      new DenseMatrix(emissionProb), new DenseVector(initialProb));
    HmmUtils.validate(model);
    writeModelToDirectory(model, modelPath, conf);
  }

  /**
   * Encodes a particular HmmModel as a Sequence File and write it to the specified location.
   *
   * @param model     HmmModel to be encoded
   * @param modelPath Location to store the encoded model
   * @param conf      Configuration object
   * @throws IOException
   */

  protected static void writeModelToDirectory(HmmModel model, Path modelPath, Configuration conf) throws IOException {

    int numHidden = model.getNrOfHiddenStates();
    int numObserved = model.getNrOfOutputStates();
    Matrix emissionMatrix = model.getEmissionMatrix();
    Matrix transitionMatrix = model.getTransitionMatrix();
    Vector initialProbability = model.getInitialProbabilities();

    MapWritable initialDistributionMap = new MapWritable();
    MapWritable transitionDistributionMap = new MapWritable();
    MapWritable emissionDistributionMap = new MapWritable();
    // delete the output directory
    HadoopUtil.delete(conf, modelPath);
    // create new file to store HMM
    FileSystem fs = FileSystem.get(modelPath.toUri(), conf);
    Path outFile = new Path(modelPath, "part-randomSeed");
    boolean newFile = fs.createNewFile(outFile);

    if (newFile) {
      SequenceFile.Writer writer = SequenceFile.createWriter(fs, conf, outFile, Text.class, MapWritable.class);

      try {
        for (int i = 0; i < numHidden; i++) {
          IntWritable initialDistributionKey = new IntWritable(i);
          DoubleWritable initialDistributionValue = new DoubleWritable(initialProbability.get(i));
          initialDistributionMap.put(initialDistributionKey, initialDistributionValue);

          Text transitionDistributionKey = new Text("TRANSIT_" + Integer.toString(i));
          MapWritable transitionDistributionValue = new MapWritable();
          for (int j = 0; j < numHidden; j++) {
            IntWritable transitionDistributionInnerKey = new IntWritable(j);
            DoubleWritable transitionDistributionInnerValue = new DoubleWritable(transitionMatrix.get(i, j));
            transitionDistributionValue.put(transitionDistributionInnerKey, transitionDistributionInnerValue);
          }
          transitionDistributionMap.put(transitionDistributionKey, transitionDistributionValue);

          Text emissionDistributionKey = new Text("EMIT_" + Integer.toString(i));
          MapWritable emissionDistributionValue = new MapWritable();
          for (int j = 0; j < numObserved; j++) {
            IntWritable emissionDistributionInnerKey = new IntWritable(j);
            DoubleWritable emissionDistributionInnerValue = new DoubleWritable(emissionMatrix.get(i, j));
            emissionDistributionValue.put(emissionDistributionInnerKey, emissionDistributionInnerValue);
          }
          emissionDistributionMap.put(emissionDistributionKey, emissionDistributionValue);
        }

        writer.append(new Text("INITIAL"), initialDistributionMap);
        log.info("Wrote random Initial Distribution Map to {}", outFile);
        for (MapWritable.Entry<Writable, Writable> transitionEntry : transitionDistributionMap.entrySet()) {

          writer.append(transitionEntry.getKey(), transitionEntry.getValue());
        }
        log.info("Wrote random Transition Distribution Map to {}", outFile);

        for (MapWritable.Entry<Writable, Writable> emissionEntry : emissionDistributionMap.entrySet()) {
          writer.append(emissionEntry.getKey(), emissionEntry.getValue());
        }
        log.info("Wrote random Emission Distribution Map to {}", outFile);

      } finally {
        Closeables.closeQuietly(writer);
      }

    }

  }

  /**
   * Checks convergence of two HMM models by computing a simple distance between
   * emission / transition matrices
   *
   * @param oldModel Old HMM Model
   * @param newModel New HMM Model
   * @param epsilon  Convergence Factor
   * @return true if training converged to a stable state.
   */
  public static boolean checkConvergence(HmmModel oldModel, HmmModel newModel,
                                         double epsilon) {
    // check convergence of transitionProbabilities
    Matrix oldTransitionMatrix = oldModel.getTransitionMatrix();
    Matrix newTransitionMatrix = newModel.getTransitionMatrix();
    double diff = 0;
    for (int i = 0; i < oldModel.getNrOfHiddenStates(); ++i) {
      for (int j = 0; j < oldModel.getNrOfHiddenStates(); ++j) {
        double tmp = oldTransitionMatrix.getQuick(i, j)
          - newTransitionMatrix.getQuick(i, j);
        diff += tmp * tmp;
      }
    }
    double norm = Math.sqrt(diff);
    diff = 0;
    // check convergence of emissionProbabilities
    Matrix oldEmissionMatrix = oldModel.getEmissionMatrix();
    Matrix newEmissionMatrix = newModel.getEmissionMatrix();
    for (int i = 0; i < oldModel.getNrOfHiddenStates(); i++) {
      for (int j = 0; j < oldModel.getNrOfOutputStates(); j++) {

        double tmp = oldEmissionMatrix.getQuick(i, j)
          - newEmissionMatrix.getQuick(i, j);
        diff += tmp * tmp;
      }
    }
    norm += Math.sqrt(diff);

    return norm < epsilon;
  }

    public static byte[] doublePairToByteArray(double d1, double d2) {
      byte[] bytes = new byte[16];
      ByteBuffer.wrap(bytes).putDouble(d1);
      ByteBuffer.wrap(bytes).putDouble(8, d2);
      return bytes;
  }
    public static double[] toDoublePair(byte[] bytes) {
	double[] pair = new double[2];
	pair[0] = ByteBuffer.wrap(bytes).getDouble();
	pair[1] = ByteBuffer.wrap(bytes).getDouble(8);
	return pair;
    }
}
