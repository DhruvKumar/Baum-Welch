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

package org.apache.mahout.classifier.sequencelearning.hmm.hadoop;

import java.io.*;

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.OptionException;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
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
import org.apache.mahout.common.CommandLineUtil;
import org.apache.mahout.common.HadoopUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Main class for launching iterative BaumWelch MapReduce training.
 * Baum-Welch, like K-Means is an Expectation Maximization (EM) algorithm which tries to estimate
 * model's parameters using maximum likelihood criterion,
 * <p/>
 * As shown by Andrew Ng et. al, EM algorithms fit nicely into the MapReduce framework.
 * <p/>
 * The training splits evenly between the mappers and reducers, with combiners reducing the network traffic.
 */

public class BaumWelchDriver extends AbstractJob {

  private static final Logger log = LoggerFactory.getLogger(BaumWelchDriver.class);

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new Configuration(), new BaumWelchDriver(), args);
  }

  @Override
  public int run(String[] args) throws Exception {


    DefaultOptionBuilder optionBuilder = new DefaultOptionBuilder();
    ArgumentBuilder argumentBuilder = new ArgumentBuilder();

    Option inputOption = optionBuilder.withLongName("input").
      withDescription("Sequence file containing VectorWritables as training sequence").
      withShortName("i").withArgument(argumentBuilder.withMaximum(1).withMinimum(1).
      withName("path").create()).withRequired(true).create();

    Option outputOption = optionBuilder.withLongName("output").
      withDescription("Output path to store the trained model encoded as Sequence Files").
      withShortName("o").withArgument(argumentBuilder.withMaximum(1).withMinimum(1).
      withName("path").create()).withRequired(true).create();

    Option modelOption = optionBuilder.withLongName("model").
      withDescription("Initial HmmModel encoded as a Sequence File. " +
        "Will be constructed with a random distribution if the 'buildRandom' option is set to true.").
      withShortName("im").withArgument(argumentBuilder.withMaximum(1).withMinimum(1).
      withName("path").create()).withRequired(false).create();

    Option hiddenStateMapPath = optionBuilder.withLongName("hiddenStateToIDMap").
      withDescription("Hidden states to ID map path.").
      withShortName("hmap").withArgument(argumentBuilder.withMaximum(1).withMinimum(1).
      withName("path").create()).withRequired(true).create();

    Option emitStateMapPath = optionBuilder.withLongName("emittedStateToIDMap").
      withDescription("Emitted states to ID map path.").
      withShortName("smap").withArgument(argumentBuilder.withMaximum(1).withMinimum(1).
      withName("path").create()).withRequired(true).create();

    Option randomOption = optionBuilder.withLongName("buildRandom").
      withDescription("Optional argument to generate a random initial HmmModel and store it in 'model' directory").
      withShortName("r").withArgument(argumentBuilder.withMaximum(1).withMinimum(1).
      withName("boolean").create()).withRequired(false).create();

    Option scalingOption = optionBuilder.withLongName("Scaling").
      withDescription("Optional argument to invoke scaled training").
      withShortName("l").withArgument(argumentBuilder.withMaximum(1).withMinimum(1).
      withName("string").create()).withRequired(true).create();

    Option stateNumberOption = optionBuilder.withLongName("nrOfHiddenStates").
      withDescription("Number of hidden states").
      withShortName("nh").withArgument(argumentBuilder.withMaximum(1).withMinimum(1).
      withName("number").create()).withRequired(true).create();

    Option observedStateNumberOption = optionBuilder.withLongName("nrOfObservedStates").
      withDescription("Number of observed states").
      withShortName("no").withArgument(argumentBuilder.withMaximum(1).withMinimum(1).
      withName("number").create()).withRequired(true).create();

    Option epsilonOption = optionBuilder.withLongName("epsilon").
      withDescription("Convergence threshold").
      withShortName("e").withArgument(argumentBuilder.withMaximum(1).withMinimum(1).
      withName("number").create()).withRequired(true).create();

    Option iterationsOption = optionBuilder.withLongName("maxIterations").
      withDescription("Maximum iterations number").
      withShortName("m").withArgument(argumentBuilder.withMaximum(1).withMinimum(1).
      withName("number").create()).withRequired(true).create();

    Group optionGroup = new GroupBuilder().withOption(inputOption).
      withOption(outputOption).withOption(modelOption).withOption(hiddenStateMapPath).
      withOption(emitStateMapPath).withOption(randomOption).withOption(scalingOption).
      withOption(stateNumberOption).withOption(observedStateNumberOption).
      withOption(epsilonOption).withOption(iterationsOption).
      withName("Options").create();

    try {
      Parser parser = new Parser();
      parser.setGroup(optionGroup);
      CommandLine commandLine = parser.parse(args);

      String input = (String) commandLine.getValue(inputOption);
      String output = (String) commandLine.getValue(outputOption);
      String modelIn = (String) commandLine.getValue(modelOption);
      String hiddenStateToIdMap = (String) commandLine.getValue(hiddenStateMapPath);
      String emittedStateToIdMap = (String) commandLine.getValue(emitStateMapPath);

      Boolean buildRandom = commandLine.hasOption(randomOption);
      String scaling = (String)commandLine.getValue(scalingOption);

      int numHidden = Integer.parseInt((String) commandLine.getValue(stateNumberOption));
      int numObserved = Integer.parseInt((String) commandLine.getValue(observedStateNumberOption));

      double convergenceDelta = Double.parseDouble((String) commandLine.getValue(epsilonOption));
      int maxIterations = Integer.parseInt((String) commandLine.getValue(iterationsOption));


      if (getConf() == null) {
        setConf(new Configuration());
      }
      if (buildRandom) {

        BaumWelchUtils.buildRandomModel(numHidden, numObserved, new Path(modelIn), getConf());
      }
      run(getConf(), new Path(input), new Path(modelIn), new Path(output),
        new Path(hiddenStateToIdMap), new Path(emittedStateToIdMap),
        numHidden, numObserved, convergenceDelta, scaling, maxIterations);
    } catch (OptionException e) {
      CommandLineUtil.printHelp(optionGroup);
    }

    return 0;

  }

  /**
   * Run the Baum-Welch Map Reduce algorithm using the supplied arguments
   *
   * @param conf                the Configuration to use
   * @param input               the Path to the directory containing input observed sequences
   * @param modelIn             the Path to the HmmModel stored as a SequenceFile
   * @param output              the Path to the output directory
   * @param hiddenStateToIdMap  the Path to the map of hidden states to ids
   * @param emittedStateToIdMap the Path to the map of emitted states to ids
   * @param numHidden           the number of Hidden states
   * @param numObserved         the number of Observed states
   * @param convergenceDelta    the convergence delta value
   * @param scaling            use the log scaled version if set to true
   * @param maxIterations       the maximum number of iterations
   * @throws IOException
   * @throws InterruptedIOException
   * @throws ClassNotFoundException
   */
  public static void run(Configuration conf,
                         Path input,
                         Path modelIn,
                         Path output,
                         Path hiddenStateToIdMap,
                         Path emittedStateToIdMap,
                         int numHidden,
                         int numObserved,
                         double convergenceDelta,
                         String scaling,
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
    Path modelOut = runBaumWelchMR(conf, input, modelIn, output, hiddenStateToIdMap, emittedStateToIdMap,
      numHidden, numObserved, delta, scaling, maxIterations);
  }

  /**
   * Run the Baum-Welch Map Reduce algorithm using the supplied arguments
   *
   * @param conf                the Configuration to use
   * @param input               the Path to the directory containing input observed sequences
   * @param modelIn             the Path to the HmmModel stored as a SequenceFile
   * @param output              the Path to the output directory
   * @param hiddenStateToIdMap  the Path to the map of hidden states to ids
   * @param emittedStateToIdMap the Path to the map of emitted states to ids
   * @param numHidden           the number of Hidden states
   * @param numObserved         the number of Observed states
   * @param delta               the convergence delta value
   * @param scaling            use the log scaled variant if set to true
   * @param maxIterations       the maximum number of iterations
   * @return the path to the output model directory
   * @throws IOException
   * @throws InterruptedIOException
   * @throws ClassNotFoundException
   */
  public static Path runBaumWelchMR(Configuration conf,
                                    Path input,
                                    Path modelIn,
                                    Path output,
                                    Path hiddenStateToIdMap,
                                    Path emittedStateToIdMap,
                                    int numHidden,
                                    int numObserved,
                                    String delta,
                                    String scaling,
                                    int maxIterations)
    throws IOException, InterruptedException, ClassNotFoundException {
    boolean converged = false;
    int iteration = 1;
    while (!converged && iteration <= maxIterations) {
      log.info("Baum-Welch MR Iteration " + iteration);
      // point the output to a new directory per iteration
      Path modelOut = new Path(output, "model-" + iteration);
      converged = runIteration(conf, input, modelIn, modelOut, hiddenStateToIdMap, emittedStateToIdMap,
        numHidden, numObserved, scaling, delta);
      modelIn = modelOut;
      iteration++;
    }
    return new Path(output.toString() + "/model-" + --iteration);
  }

  /**
   * Run one iteration of the Baum-Welch Map Reduce algorithm using the supplied arguments
   *
   * @param conf                the Configuration to use
   * @param input               the Path to the directory containing input
   * @param modelIn             the Path to the HmmModel
   * @param modelOut            the Path to the output directory
   * @param hiddenStateToIdMap  the Path to the map of hidden states to ids
   * @param emittedStateToIdMap the Path to the map of emitted states to ids
   * @param numHidden           the number of Hidden states
   * @param numObserved         the number of Observed states
   * @param scaling             name of the scaling method
   * @param delta               the convergence delta value
   * @return true or false depending on convergence check
   */

  private static boolean runIteration(Configuration conf,
                                      Path input,
                                      Path modelIn,
                                      Path modelOut,
                                      Path hiddenStateToIdMap,
                                      Path emittedStateToIdMap,
                                      int numHidden,
                                      int numObserved,
                                      String scaling,
                                      String delta)
    throws IOException, InterruptedException, ClassNotFoundException {

    conf.set(BaumWelchConfigKeys.EMITTED_STATES_MAP_PATH, emittedStateToIdMap.toString());
    conf.set(BaumWelchConfigKeys.HIDDEN_STATES_MAP_PATH, hiddenStateToIdMap.toString());
    conf.set(BaumWelchConfigKeys.SCALING_OPTION_KEY, scaling);
    conf.set(BaumWelchConfigKeys.MODEL_PATH_KEY, modelIn.toString());
    conf.set(BaumWelchConfigKeys.NUMBER_OF_HIDDEN_STATES_KEY, ((Integer) numHidden).toString());
    conf.set(BaumWelchConfigKeys.NUMBER_OF_EMITTED_STATES_KEY, ((Integer) numObserved).toString());
    conf.set(BaumWelchConfigKeys.MODEL_CONVERGENCE_KEY, delta);

    Job job = new Job(conf, "Baum-Welch Driver running runIteration over modelIn: " + conf.get(BaumWelchConfigKeys.MODEL_PATH_KEY));
    job.setMapOutputKeyClass(Text.class);
    job.setMapOutputValueClass(MapWritable.class);
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(MapWritable.class);

    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    job.setMapperClass(BaumWelchMapper.class);
    job.setCombinerClass(BaumWelchCombiner.class);
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
   * @return true if converged, false otherwise
   */

  private static boolean isConverged(Path modelIn, Path modelOut, int numHidden, int numObserved, Configuration conf) throws IOException {

    log.info("-----------Checking Convergence----------");
    HmmModel previousModel = BaumWelchUtils.createHmmModel(numHidden, numObserved, modelIn, conf);
    if (previousModel == null) {
      throw new IllegalStateException("HmmModel from previous iteration is empty!");
    }
    HmmModel newModel = BaumWelchUtils.createHmmModel(numHidden, numObserved, modelOut, conf);
    if (newModel == null) {
      throw new IllegalStateException("HmmModel from current iteration is empty!");
    }

    return BaumWelchUtils.checkConvergence(previousModel, newModel, Double.parseDouble(conf.get(BaumWelchConfigKeys.MODEL_CONVERGENCE_KEY)));

  }


}
