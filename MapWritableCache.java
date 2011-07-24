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
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.mahout.common.HadoopUtil;

import java.io.IOException;
import java.net.URI;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileValueIterator;

/**
 * This class handles reading and writing MapWritable to the Hadoop
 * distributed cache.
 */
public final class MapWritableCache {

  private MapWritableCache() {
  }

  /**
   * @param key SequenceFile key
   * @param map Map to save
   */
  public static void save(Writable key,
                          MapWritable map,
                          Path output,
                          Configuration conf,
                          boolean overwritePath,
                          boolean deleteOnExit) throws IOException {

    FileSystem fs = FileSystem.get(conf);
    output = fs.makeQualified(output);
    if (overwritePath) {
      HadoopUtil.delete(conf, output);
    }

    // set the cache
    DistributedCache.setCacheFiles(new URI[]{output.toUri()}, conf);

    // set up the writer
    SequenceFile.Writer writer = new SequenceFile.Writer(fs, conf, output,
      IntWritable.class, MapWritable.class);
    try {
      writer.append(key, new MapWritable(map));
    } finally {
      Closeables.closeQuietly(writer);
    }

    if (deleteOnExit) {
      fs.deleteOnExit(output);
    }
  }

  /**
   * Calls the save() method, setting the cache to overwrite any previous
   * Path and to delete the path after exiting
   */
  public static void save(Writable key, MapWritable map, Path output, Configuration conf) throws IOException {
    save(key, map, output, conf, true, true);
  }

  /**
   * Loads a MapWritable from {@link DistributedCache}. Returns null if no map exists.
   */
  public static MapWritable load(Configuration conf) throws IOException {
    URI[] files = DistributedCache.getCacheFiles(conf);
    if (files == null || files.length < 1) {
      return null;
    }
    return load(conf, new Path(files[0].getPath()));
  }

  /**
   * Loads a MapWritable from the specified path. Returns null if no map exists.
   */
  public static MapWritable load(Configuration conf, Path input) throws IOException {
    SequenceFileValueIterator<MapWritable> iterator =
      new SequenceFileValueIterator<MapWritable>(input, true, conf);
    try {
      return iterator.next();
    } finally {
      Closeables.closeQuietly(iterator);
    }
  }

}
