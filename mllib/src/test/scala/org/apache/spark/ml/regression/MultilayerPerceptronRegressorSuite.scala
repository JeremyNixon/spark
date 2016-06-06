/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.ml.regression

import org.apache.spark.SparkFunSuite
import org.apache.spark.ml.classification.LogisticRegressionSuite._
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.regression.MultilayerPerceptronRegressor
import org.apache.spark.ml.util.DefaultReadWriteTest
import org.apache.spark.ml.util.MLTestingUtils
import org.apache.spark.ml.util.TestingUtils._
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.{Vectors => OldVectors}
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.sql.{Dataset, Row}

class MultilayerPerceptronRegressorSuite
  extends SparkFunSuite with MLlibTestSparkContext {

  test("MLPRegressor behaves reasonably on toy data") {

    val df = spark.createDataFrame(Seq(
      LabeledPoint(10, Vectors.dense(1, 2, 3, 4)),
      LabeledPoint(-5, Vectors.dense(6, 3, 2, 1)),
      LabeledPoint(11, Vectors.dense(2, 2, 3, 4)),
      LabeledPoint(-6, Vectors.dense(6, 4, 2, 1)),
      LabeledPoint(9, Vectors.dense(1, 2, 6, 4)),
      LabeledPoint(-4, Vectors.dense(6, 3, 2, 2))
    ))
    val mlpr = new MultilayerPerceptronRegressor().setLayers(Array[Int](4, 10, 10, 1))
    val model = mlpr.fit(df)
    val results = model.transform(df)
    val predictions = results.select("prediction").rdd.map(_.getDouble(0))
    assert(predictions.max() > 2)
    assert(predictions.min() < -1)
  }

  test("Input Validation") {
    val mlpr = new MultilayerPerceptronRegressor()
    intercept[IllegalArgumentException] {
      mlpr.setLayers(Array[Int]())
    }
    intercept[IllegalArgumentException] {
      mlpr.setLayers(Array[Int](1))
    }
    intercept[IllegalArgumentException] {
      mlpr.setLayers(Array[Int](0, 1))
    }
    intercept[IllegalArgumentException] {
      mlpr.setLayers(Array[Int](1, 0))
    }
    mlpr.setLayers(Array[Int](1, 1))
  }

  test("Test setWeights by training restart") {
    val dataFrame = spark.createDataFrame(Seq(
      LabeledPoint(10, Vectors.dense(1, 2, 3, 4)),
      LabeledPoint(-5, Vectors.dense(6, 3, 2, 1)),
      LabeledPoint(11, Vectors.dense(2, 2, 3, 4)),
      LabeledPoint(-6, Vectors.dense(6, 4, 2, 1)),
      LabeledPoint(9, Vectors.dense(1, 2, 6, 4)),
      LabeledPoint(-4, Vectors.dense(6, 3, 2, 2))
    ))
    val layers = Array[Int](2, 5, 2)
    val trainer = new MultilayerPerceptronRegressor()
      .setLayers(layers)
      .setBlockSize(1)
      .setSeed(12L)
      .setMaxIter(1)
      .setTol(1e-6)
    val initialWeights = trainer.fit(dataFrame).weights
    trainer.setInitialWeights(initialWeights.copy)
    val weights1 = trainer.fit(dataFrame).weights
    trainer.setInitialWeights(initialWeights.copy)
    val weights2 = trainer.fit(dataFrame).weights
    assert(weights1 ~== weights2 absTol 10e-5,
      "Training should produce the same weights given equal initial weights and number of steps")
  }

  /* Test for numeric types after rewriting max/min for Dataframe method to handle Long/BigInt */

}
