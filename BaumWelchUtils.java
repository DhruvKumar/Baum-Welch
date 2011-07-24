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

import com.google.common.io.Closeables;
import org.apache.commons.lang.ArrayUtils;
import org.apache.hadoop.classification.InterfaceAudience;
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
import org.jfree.util.Log;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;


public class BaumWelchUtils {

  private static final Logger log = LoggerFactory.getLogger(BaumWelchUtils.class);

  private BaumWelchUtils() {

  }

  public static HmmModel CreateHmmModel(int nrOfHiddenStates,
                                        int nrOfOutputStates,
                                        Path modelPath,
                                        Configuration conf) throws IOException {


    log.info("Entering Create Hmm Model. Model Path = {}", modelPath.toUri());
    Vector initialProbabilities = new DenseVector(nrOfHiddenStates);
    Matrix transitionMatrix = new DenseMatrix(nrOfHiddenStates, nrOfHiddenStates);
    Matrix emissionMatrix = new DenseMatrix(nrOfHiddenStates, nrOfOutputStates);

    // Get the path location where the seq files encoding model are stored
    Path modelFilesPath = new Path(modelPath, "*");
    log.info("Create Hmm Model. ModelFiles Path = {}", modelFilesPath.toUri());
    Collection<Path> result = new ArrayList<Path>();

    // get all filtered file names in result list
    FileSystem fs = modelFilesPath.getFileSystem(conf);
    log.info("Create Hmm Model. File System = {}", fs);
    FileStatus[] matches = fs.listStatus(FileUtil.stat2Paths(fs.globStatus(modelFilesPath, PathFilters.partFilter())),
      PathFilters.partFilter());

    for (FileStatus match : matches) {
      log.info("CreateHmmmModel Adding File Match {}", match.getPath().toString());
      result.add(fs.makeQualified(match.getPath()));
    }

    // iterate through the result path list
    for (Path path : result) {
      for (Pair<Writable, MapWritable> pair : new SequenceFileIterable<Writable, MapWritable>(path, true, conf)) {
        Text key = (Text) pair.getFirst();
        log.info("CreateHmmModel Matching Seq File Key = {}", key);
        MapWritable valueMap = pair.getSecond();
        if (key.charAt(0) == 'I') {
          // initial distribution stripe
          for (MapWritable.Entry<Writable, Writable> entry : valueMap.entrySet()) {
            log.info("CreateHmmModel Initial Prob Adding  Key, Value  = ({} {})",
              ((IntWritable) entry.getKey()).get(), ((DoubleWritable) entry.getValue()).get());
            initialProbabilities.set(((IntWritable) entry.getKey()).get(), ((DoubleWritable) entry.getValue()).get());
          }
        } else if (key.charAt(0) == 'T') {
          // transition distribution stripe
          // key is of the form TRANSIT_0, TRANSIT_1 etc
          // the number after _ is the state ID at char number 11
          int stateID = Character.getNumericValue(key.charAt(8));
          log.info("CreateHmmModel stateID = key.charAt(8) = {}", stateID);
          for (MapWritable.Entry<Writable, Writable> entry : valueMap.entrySet()) {
            log.info("CreateHmmModel Transition Matrix ({}, {}) = {}",
              new Object[]{stateID, ((IntWritable) entry.getKey()).get(), ((DoubleWritable) entry.getValue()).get()});
            transitionMatrix.set(stateID, ((IntWritable) entry.getKey()).get(), ((DoubleWritable) entry.getValue()).get());
          }
        } else if (key.charAt(0) == 'E') {
          // emission distribution stripe
          // key is of the form EMIT_0, EMIT_1 etc
          // the number after _ is the state ID at char number 5
          int stateID = Character.getNumericValue(key.charAt(5));
          for (MapWritable.Entry<Writable, Writable> entry : valueMap.entrySet()) {
            log.info("CreateHmmModel Emission Matrix ({}, {}) = {}",
              new Object[]{stateID, ((IntWritable) entry.getKey()).get(), ((DoubleWritable) entry.getValue()).get()});
            emissionMatrix.set(stateID, ((IntWritable) entry.getKey()).get(), ((DoubleWritable) entry.getValue()).get());
          }
        } else {
          throw new IllegalStateException("Error creating HmmModel from Sequence File Path");
        }
      }
    }
    HmmModel model = new HmmModel(transitionMatrix, emissionMatrix, initialProbabilities);
    HmmUtils.validate(model);
    return model;
  }

  public static void BuildRandomModel(int numHidden,
                                      int numObserved,
                                      Path modelPath,
                                      Configuration conf) throws IOException {
    HmmModel model = new HmmModel(numHidden, numObserved);
    HmmUtils.validate(model);
    WriteModelToDirectory(model, modelPath, conf);
  }

  public static void BuildHmmModelFromDistributions(double[] initialProb,
                                                    double[][] transitionProb,
                                                    double[][] emissionProb,
                                                    Path modelPath,
                                                    Configuration conf) throws IOException {
    HmmModel model = new HmmModel(new DenseMatrix(transitionProb),
      new DenseMatrix(emissionProb), new DenseVector(initialProb));
    HmmUtils.validate(model);
    WriteModelToDirectory(model, modelPath, conf);
  }

  protected static void WriteModelToDirectory(HmmModel model, Path modelPath, Configuration conf) throws IOException {

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

        // construct one MapWritable<IntWritable, DoubleWritable> object
        // and two MapWritable<Text, MapWritable<IntWritable, DoubleWritable >> objects
        for (int i = 0; i < numHidden; i++) {
          IntWritable initialDistributionKey = new IntWritable(i);
          DoubleWritable initialDistributionValue = new DoubleWritable(initialProbability.get(i));
          log.info("BuildRandomModel Initial Distribution Map: State {} = {})",
            initialDistributionKey.get(), initialDistributionValue.get());
          initialDistributionMap.put(initialDistributionKey, initialDistributionValue);

          Text transitionDistributionKey = new Text("TRANSIT_" + Integer.toString(i));
          MapWritable transitionDistributionValue = new MapWritable();
          for (int j = 0; j < numHidden; j++) {
            IntWritable transitionDistributionInnerKey = new IntWritable(j);
            DoubleWritable transitionDistributionInnerValue = new DoubleWritable(transitionMatrix.get(i, j));
            log.info("BuildRandomModel Transition Distribution Map Inner: ({}, {}) = ({}, {})",
              new Object[]{i, j, transitionDistributionInnerKey.get(), transitionDistributionInnerValue.get()});
            transitionDistributionValue.put(transitionDistributionInnerKey, transitionDistributionInnerValue);
          }
          transitionDistributionMap.put(transitionDistributionKey, transitionDistributionValue);

          Text emissionDistributionKey = new Text("EMIT_" + Integer.toString(i));
          MapWritable emissionDistributionValue = new MapWritable();
          for (int j = 0; j < numObserved; j++) {
            IntWritable emissionDistributionInnerKey = new IntWritable(j);
            DoubleWritable emissionDistributionInnerValue = new DoubleWritable(emissionMatrix.get(i, j));
            log.info("BuildRandomModel Emission Distribution Map Inner: ({}, {}) = ({}, {})",
              new Object[]{i, j, emissionDistributionInnerKey.get(), emissionDistributionInnerValue.get()});
            emissionDistributionValue.put(emissionDistributionInnerKey, emissionDistributionInnerValue);
          }
          emissionDistributionMap.put(emissionDistributionKey, emissionDistributionValue);

        }

        writer.append(new Text("INITIAL"), initialDistributionMap);
        log.info("Wrote random Initial Distribution Map to {}", outFile);

        for (MapWritable.Entry<Writable, Writable> transitionEntry : transitionDistributionMap.entrySet()) {
          log.info("Writing Transition Distribution Map Key, Value = ({}, {})",
            transitionEntry.getKey(), transitionEntry.getValue());
          writer.append(transitionEntry.getKey(), transitionEntry.getValue());
        }
        log.info("Wrote random Transition Distribution Map to {}", outFile);

        for (MapWritable.Entry<Writable, Writable> emissionEntry : emissionDistributionMap.entrySet()) {
          log.info("Writing Emission Distribution Map Key, Value = ({}, {})",
            emissionEntry.getKey(), emissionEntry.getValue());
          writer.append(emissionEntry.getKey(), emissionEntry.getValue());
        }
        log.info("Wrote random Emission Distribution Map to {}", outFile);

      } finally {
        Closeables.closeQuietly(writer);
      }

    }

  }

  /**
   * Check convergence of two HMM models by computing a simple distance between
   * emission / transition matrices
   *
   * @param oldModel Old HMM Model
   * @param newModel New HMM Model
   * @param epsilon  Convergence Factor
   * @return true if training converged to a stable state.
   */
  public static boolean CheckConvergence(HmmModel oldModel, HmmModel newModel,
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
    // iteration has converged :)
    return norm < epsilon;
  }
}
