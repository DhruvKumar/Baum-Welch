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
import java.util.Map;


import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Reducer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Finishes the Expectation step started by mappers, computes the model parameters for next iteration
 * and writes them to the distributed file system.
 */

public class BaumWelchReducer extends Reducer<Text, MapWritable, Text, MapWritable> {

  private static final Logger log = LoggerFactory.getLogger(BaumWelchReducer.class);
  private String scaling = "noscaling";

  @Override
  public void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    Configuration config = context.getConfiguration();
    scaling = (String)config.get(BaumWelchConfigKeys.SCALING_OPTION_KEY);
  }

  @Override
  protected void reduce(Text key, Iterable<MapWritable> stripes, Context context) throws IOException, InterruptedException {

    MapWritable sumOfStripes = new MapWritable();

    // Finish the Expectation Step by aggregating all posterior probabilities for one key
    if (scaling.equals("logscaling")) {
      double totalValSum = Double.NEGATIVE_INFINITY;
      for (MapWritable stripe : stripes) {
        for (Map.Entry e : stripe.entrySet()) {
          double val = ((DoubleWritable) e.getValue()).get();
          double max = totalValSum > val ? totalValSum : val;
          totalValSum = max + Math.log(Math.exp(totalValSum - max) + Math.exp(val - max));
          if (!sumOfStripes.containsKey(e.getKey())) {
            sumOfStripes.put((IntWritable) e.getKey(), new DoubleWritable(val));
          } else {
            double sumSripesVal = ((DoubleWritable) sumOfStripes.get(e.getKey())).get();
            if (sumSripesVal > Double.NEGATIVE_INFINITY) {
              val = val + Math.log(1 + Math.exp(sumSripesVal - val));
            }
            sumOfStripes.put((IntWritable) e.getKey(), new DoubleWritable(val));
          }
        }
      }

      //normalize the aggregate
      for (Map.Entry e : sumOfStripes.entrySet()) {
        double val = ((DoubleWritable) e.getValue()).get();
        if (totalValSum > Double.NEGATIVE_INFINITY) {
          val = val - totalValSum;
        }
        sumOfStripes.put((IntWritable) e.getKey(), new DoubleWritable(Math.exp(val)));
      }
    } else if (scaling.equals("rescaling")) {
	double totalValSum = 0.0;

		for (MapWritable stripe : stripes) {
				for (Map.Entry e : stripe.entrySet()) {
					if (key.charAt(0) == (int) 'I') {
						double val = ((DoubleWritable) e.getValue()).get();
						totalValSum += val;
						if (!sumOfStripes.containsKey(e.getKey())) {
							sumOfStripes.put((IntWritable) e.getKey(), (DoubleWritable) e.getValue());
						} else {
							val += ((DoubleWritable) sumOfStripes.get(e.getKey())).get();
							sumOfStripes.put((IntWritable) e.getKey(), new DoubleWritable(val));
						}
					} else {
double[] pr = BaumWelchUtils.toDoublePair(((BytesWritable)e.getValue()).getBytes());
double num = pr[0];
double denom = pr[1];
						if (!sumOfStripes.containsKey(e.getKey())) {
						    sumOfStripes.put((IntWritable) e.getKey(), (BytesWritable)e.getValue());
						} else {
						    double[] pr1 = BaumWelchUtils.toDoublePair(((BytesWritable) sumOfStripes.get(e.getKey())).getBytes());
						    num += pr1[0];
						    denom += pr1[1];
							byte [] doublePair1 = BaumWelchUtils.doublePairToByteArray(num, denom);
							sumOfStripes.put((IntWritable) e.getKey(), new BytesWritable(doublePair1));
						}
					}
				}
			}
		
		if (key.charAt(0) == (int) 'I') {
			//normalize the aggregate
			for (Map.Entry e : sumOfStripes.entrySet()) {
				double val = ((DoubleWritable) e.getValue()).get();
				if (totalValSum > 0) {
					val /= totalValSum;
				}
				sumOfStripes.put((IntWritable) e.getKey(), new DoubleWritable(val));
			}
			
		} else {
			// compute the probabilities
			for (Map.Entry e : sumOfStripes.entrySet()) {
				double[] pr1 = BaumWelchUtils.toDoublePair(((BytesWritable) sumOfStripes.get(e.getKey())).getBytes());
				sumOfStripes.put((IntWritable) e.getKey(), new DoubleWritable(pr1[0]/pr1[1]));
			}
		}
    } else {
      double totalValSum = 0.0;

      for (MapWritable stripe : stripes) {
        for (Map.Entry e : stripe.entrySet()) {
          int state = ((IntWritable) e.getKey()).get();
          double val = ((DoubleWritable) e.getValue()).get();
          totalValSum += val;
          if (!sumOfStripes.containsKey(e.getKey())) {
            sumOfStripes.put((IntWritable) e.getKey(), (DoubleWritable) e.getValue());
          } else {
            val += ((DoubleWritable) sumOfStripes.get(e.getKey())).get();
            sumOfStripes.put((IntWritable) e.getKey(), new DoubleWritable(val));
          }
        }
      }

      //normalize the aggregate
      for (Map.Entry e : sumOfStripes.entrySet()) {
        double val = ((DoubleWritable) e.getValue()).get();
        if (totalValSum > 0) {
          val /= totalValSum;
        }
        sumOfStripes.put((IntWritable) e.getKey(), new DoubleWritable(val));
      }
    }

    //Write the distribution parameter vector to HDFS for the next iteration
    context.write(key, sumOfStripes);

  }
}

