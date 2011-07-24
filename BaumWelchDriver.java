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

import java.io.*;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.ToolRunner;

import org.apache.mahout.classifier.sequencelearning.hmm.HmmModel;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class BaumWelchDriver extends AbstractJob {

  private static final Logger log = LoggerFactory.getLogger(BaumWelchDriver.class);

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new Configuration(), new BaumWelchDriver(), args);
  }

  @Override
  public int run(String[] args) throws Exception {

    addInputOption();
    addOutputOption();
    addOption(DefaultOptionCreator.modelInOption()
      .withDescription("The input HMM Model. Must be of Sequence File type.")
      .create());
    addOption(DefaultOptionCreator.numHiddenStatesOption().create());
    addOption(DefaultOptionCreator.numObservedStatesOption().create());
    addOption(DefaultOptionCreator.convergenceOption().create());
    addOption(DefaultOptionCreator.maxIterationsOption().create());
    if (parseArguments(args) == null) {
      return -1;
    }
    Path input = getInputPath();
    Path modelIn = new Path(DefaultOptionCreator.HMM_IN_OPTION);
    Path output = getOutputPath();
    int numHidden = Integer.parseInt(getOption(DefaultOptionCreator.NUM_HIDDEN_STATES));
    int numObserved = Integer.parseInt(getOption(DefaultOptionCreator.NUM_OBSERVED_STATES));
    double convergenceDelta = Double.parseDouble(getOption(DefaultOptionCreator.CONVERGENCE_DELTA_OPTION));
    int maxIterations = Integer.parseInt(getOption(DefaultOptionCreator.MAX_ITERATIONS_OPTION));
    boolean buildRandom = hasOption(DefaultOptionCreator.GENERATE_RANDOM_HMM_OPTION);
    if (getConf() == null) {
      setConf(new Configuration());
    }
    if (buildRandom) {
      BaumWelchUtils.BuildRandomModel(numHidden, numObserved, modelIn, getConf());
    }
    run(getConf(), input, modelIn, output, numHidden, numObserved, convergenceDelta, maxIterations);
    return 0;
  }

  /**
   * Run the Baum-Welch Map Reduce algorithm using the supplied arguments
   *
   * @param conf             the Configuration to use
   * @param input            the Path to the directory containing input
   * @param modelIn          the Path to the HmmModel
   * @param output           the Path to the output directory
   * @param numHidden        the number of Hidden states
   * @param numObserved      the number of Observed states
   * @param convergenceDelta the convergence delta value
   * @param maxIterations    the maximum number of iterations
   */
  public static void run(Configuration conf,
                         Path input,
                         Path modelIn,
                         Path output,
                         int numHidden,
                         int numObserved,
                         double convergenceDelta,
                         int maxIterations)
    throws IOException, InterruptedException, ClassNotFoundException {

    //iterate until the model converges or until maxIterations
    String delta = Double.toString(convergenceDelta);
    if (log.isInfoEnabled()) {
      log.info("Input: {} Model In: {} Out: {}",
        new Object[]{input, modelIn, output});
      log.info("convergence: {} max Iterations: {}",
        new Object[]{convergenceDelta, maxIterations});
    }
    Path modelOut = runBaumWelchMR(conf, input, modelIn, output, numHidden, numObserved, delta, maxIterations);
  }

  private static Path runBaumWelchMR(Configuration conf,
                                     Path input,
                                     Path modelIn,
                                     Path output,
                                     int numHidden,
                                     int numObserved,
                                     String delta,
                                     int maxIterations)
    throws IOException, InterruptedException, ClassNotFoundException {
    boolean converged = false;
    int iteration = 1;
    while (!converged && iteration <= maxIterations) {
      log.info("Baum-Welch MR Iteration {} " + iteration);
      // point the output to a new directory per iteration
      Path modelOut = new Path(output, "model-" + iteration);
      converged = runIteration(conf, input, modelIn, modelOut, numHidden, numObserved, delta);
      modelIn = modelOut;
      iteration++;
    }
    return modelIn;
  }

  /**
   * Run one iteration of the Baum-Welch Map Reduce algorithm using the supplied arguments
   *
   * @param conf        the Configuration to use
   * @param input       the Path to the directory containing input
   * @param modelIn     the Path to the HmmModel
   * @param modelOut    the Path to the output directory
   * @param numHidden   the number of Hidden states
   * @param numObserved the number of Observed states
   * @param delta       the convergence delta value
   */

  private static boolean runIteration(Configuration conf,
                                      Path input,
                                      Path modelIn,
                                      Path modelOut,
                                      int numHidden,
                                      int numObserved,
                                      String delta)
    throws IOException, InterruptedException, ClassNotFoundException {

    conf.set(BaumWelchConfigKeys.MODEL_PATH_KEY, modelIn.toString());
    conf.set(BaumWelchConfigKeys.NUMBER_OF_HIDDEN_STATES_KEY, ((Integer) numHidden).toString());
    conf.set(BaumWelchConfigKeys.NUMBER_OF_EMITTED_STATES_KEY, ((Integer) numObserved).toString());
    conf.set(BaumWelchConfigKeys.MODEL_CONVERGENCE_KEY, delta);

    Job job = new Job(conf, "Baum-Welch Driver running runIteration over modelIn: " + modelIn);
    job.setMapOutputKeyClass(Text.class);
    job.setMapOutputValueClass(MapWritable.class);
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(MapWritable.class);

    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    job.setMapperClass(BaumWelchMapper.class);
    //job.setCombinerClass(BaumWelchCombiner.class);
    job.setReducerClass(BaumWelchReducer.class);

    FileInputFormat.addInputPath(job, input);
    FileOutputFormat.setOutputPath(job, modelOut);

    job.setJarByClass(BaumWelchDriver.class);
    HadoopUtil.delete(conf, modelOut);
    if (!job.waitForCompletion(true)) {
      throw new InterruptedException("Baum-Welch Iteration failed processing " + modelIn);
    }

    return isConverged(modelIn, modelOut, numHidden, numObserved, conf);
  }

  /**
   * Check for convergence of two Hmm models
   *
   * @param modelIn     the Path to the HmmModel
   * @param modelOut    the Path to the output directory
   * @param numHidden   the number of Hidden states
   * @param numObserved the number of Observed states
   * @param conf        the Configuration to use
   */

  private static boolean isConverged(Path modelIn, Path modelOut, int numHidden, int numObserved, Configuration conf) throws IOException {
    HmmModel previousModel = BaumWelchUtils.CreateHmmModel(numHidden, numObserved, modelIn, conf);
    if (previousModel == null) {
      throw new IllegalStateException("HmmModel from previous iteration is empty!");
    }
    HmmModel newModel = BaumWelchUtils.CreateHmmModel(numHidden, numObserved, modelOut, conf);
    if (newModel == null) {
      throw new IllegalStateException("HmmModel from current iteration is empty!");
    }
    return BaumWelchUtils.CheckConvergence(previousModel, newModel, Double.parseDouble(conf.get(BaumWelchConfigKeys.MODEL_CONVERGENCE_KEY)));

  }


}
