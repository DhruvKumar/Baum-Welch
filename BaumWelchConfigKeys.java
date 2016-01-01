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

/**
 * This class holds all config keys that are relevant to be used in the Baum-Welch MapReduce configuration.
 */

public interface BaumWelchConfigKeys {

  /**
   * Configuration key to indicate if log scaled version is to be used
   */
  String SCALING_OPTION_KEY = "org.apache.mahout.classifier.sequencelearning.baumwelchmapreduce.scaling";

  /**
   * Configuration key to store the number of hidden states
   */
  String NUMBER_OF_HIDDEN_STATES_KEY = "org.apache.mahout.classifier.sequencelearning.baumwelchmapreduce.nrofhiddenstates";

  /**
   * Configuration key to store the number of emitted states
   */
  String NUMBER_OF_EMITTED_STATES_KEY = "org.apache.mahout.classifier.sequencelearning.baumwelchmapreduce.nrofemittedstates";

  /**
   * Configuration key to store the path of the model used in the current iteration
   */
  String MODEL_PATH_KEY = "org.apache.mahout.classifier.sequencelearning.baumwelchmapreduce.modelpath";

  /**
   * Configuration key to store the covergence delta value
   */
  String MODEL_CONVERGENCE_KEY = "org.apache.mahout.classifier.sequencelearning.baumwelchmapreduce.comvergence";

  /**
   * Configuration key to store the mapping of hidden state IDs to integers
   */
  String HIDDEN_STATES_MAP_PATH = "org.apache.mahout.classifier.sequencelearning.baumwelchmapreduce.hiddenstatesmap";

  /**
   * Configuration key to store the mapping of emitted state IDs to integers
   */
  String EMITTED_STATES_MAP_PATH = "org.apache.mahout.classifier.sequencelearning.baumwelchmapreduce.emittedstatesmap";
}
