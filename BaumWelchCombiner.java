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
import java.util.Iterator;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Reducer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class BaumWelchCombiner extends Reducer<Text, MapWritable, Text, MapWritable> {

  private static final Logger log = LoggerFactory.getLogger(BaumWelchCombiner.class);
  private Integer nrOfHiddenStates;
  private Integer nrOfEmittedStates;
  private Path modelPath;

  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    Configuration config = context.getConfiguration();
    nrOfHiddenStates = Integer.parseInt(config.get(BaumWelchConfigKeys.NUMBER_OF_HIDDEN_STATES_KEY));
    nrOfEmittedStates = Integer.parseInt(config.get(BaumWelchConfigKeys.NUMBER_OF_EMITTED_STATES_KEY));
    modelPath = new Path(BaumWelchConfigKeys.MODEL_PATH_KEY);
  }

  @Override
  protected void reduce(Text key,
                        Iterable<MapWritable> stripes,
                        Context context) throws IOException, InterruptedException {

    log.info("Entering Reducer. Key = {}", key.toString());
    MapWritable sumOfStripes = new MapWritable();
    MapWritable finalStripe = new MapWritable();
    boolean isInitial = false;
    boolean isTransit = false;
    boolean isEmit = false;


    if (key.charAt(0) == 'I') {
      isInitial = true;
    } else if (key.charAt(0) == 'E') {
      isEmit = true;
    } else if (key.charAt(0) == 'T') {
      isTransit = true;
    } else {
      throw new IllegalStateException("Baum Welch Reducer Error Determining the Key Type");
    }

    if (isInitial) {
      Double[] val = new Double[nrOfHiddenStates];
      for (int i = 0; i < nrOfHiddenStates; i++) {
        val[i] = 0.0;
      }
      for (MapWritable stripe : stripes) {
        log.info("Reducer Processing Initial Distribution Stripe.");
        for (MapWritable.Entry<Writable, Writable> stripeEntry : stripe.entrySet()) {
          log.info("Reducer Getting Initial Distribution Stripe Entry. Key = {}  Value = {} ",
            Integer.toString(((IntWritable) stripeEntry.getKey()).get()),
            Double.toString(((DoubleWritable) stripeEntry.getValue()).get()));
          val[((IntWritable) stripeEntry.getKey()).get()] += ((DoubleWritable) stripeEntry.getValue()).get();
        }
      }
      for (int i = 0; i < nrOfHiddenStates; i++) {
        log.info("Reducer adding to sumOfStripes for Initial. Key = {}  Value ={}",
          Integer.toString(i), Double.toString(val[i]));
        sumOfStripes.put(new IntWritable(i), new DoubleWritable(val[i]));
      }
    } else if (isEmit) {
      Iterator<MapWritable> it = stripes.iterator();
      int seqlength = it.next().size();
      Double[] val = new Double[nrOfEmittedStates];
      for (int i = 0; i < nrOfEmittedStates; i++) {
        val[i] = 0.0;
      }
      for (MapWritable stripe : stripes) {
        log.info("Reducer Processing Emission Distribution Stripe.");
        for (MapWritable.Entry<Writable, Writable> stripeEntry : stripe.entrySet()) {
          log.info("Reducer Getting Emission Distribution Stripe Entry. Key = {}  Value = {} ",
            Integer.toString(((IntWritable) stripeEntry.getKey()).get()),
            Double.toString(((DoubleWritable) stripeEntry.getValue()).get()));
          val[((IntWritable) stripeEntry.getKey()).get()] += ((DoubleWritable) stripeEntry.getValue()).get();
        }
      }
      for (int i = 0; i < nrOfEmittedStates; i++) {
        log.info("Reducer adding to sumOfStripes for Emission. Key = {}  Value ={}",
          Integer.toString(i), Double.toString(val[i]));
        sumOfStripes.put(new IntWritable(i), new DoubleWritable(val[i]));
      }
    } else if (isTransit) {
      Double[] val = new Double[nrOfHiddenStates];
      for (int i = 0; i < nrOfHiddenStates; i++) {
        val[i] = 0.0;
      }
      for (MapWritable stripe : stripes) {
        log.info("Reducer Processing Transition Distribution Stripe.");
        for (MapWritable.Entry<Writable, Writable> stripeEntry : stripe.entrySet()) {
          log.info("Reducer Getting Transition Distribution Stripe Entry. Key = {}  Value = {} ",
            Integer.toString(((IntWritable) stripeEntry.getKey()).get()),
            Double.toString(((DoubleWritable) stripeEntry.getValue()).get()));
          val[((IntWritable) stripeEntry.getKey()).get()] += ((DoubleWritable) stripeEntry.getValue()).get();
        }
      }
      for (int i = 0; i < nrOfHiddenStates; i++) {
        log.info("Reducer adding to sumOfStripes for Transition. Key = {}  Value ={}", Integer.toString(i), Double.toString(val[i]));
        sumOfStripes.put(new IntWritable(i), new DoubleWritable(val[i]));
      }
    } else {
      throw new IllegalStateException("Baum Welch Reducer Error: Unable to aggregate distribution stripes.");
    }

    context.write(key, sumOfStripes);

  }
}