package com.github.nmasahiro.kf

import java.io.{File, PrintWriter}

import breeze.linalg.{DenseMatrix, DenseVector}
import com.github.nmasahiro.util.Using
import com.typesafe.config.ConfigFactory

object KalmanFilterApp extends App {

  val conf = ConfigFactory.load()
  val T = conf.getInt("kf.common.sim-num")
  val xDim = conf.getInt("kf.common.x-dim")
  val yDim = conf.getInt("kf.common.y-dim")

  // initial distribution
  val μ0 = DenseVector.zeros[Double](xDim)
  val Σ0 = DenseMatrix.eye[Double](xDim)

  val x0 = DenseVector(0.0, 0.0)
  // state model
  val F = DenseMatrix((0.0, -0.7), (1.0, -1.5))
  val G = DenseMatrix(0.5, 1.0) // note that the shape of this matrix is (2, 1)
  val H = DenseMatrix(0.0, 1.0).t

  // noise parameter
  // system noise covariance
  val Q = DenseMatrix(1.0)
  // observation noise covariance
  val R = DenseMatrix(0.3)

  val model = StateSpaceModel(F, G, H, Q, R)

  val xAndy = StateAndObs(model, x0, T)
  val (xTrue, y) = (xAndy.map(_._1), xAndy.map(_._2))

  val driver = KFDriver(μ0, Σ0, y, model,
    iterationsExceed(T)
//      orElse maxEigConverged(1e-4)
      orElse proceed
  )

  val distributions = driver.estimate()
  val μs = distributions.map(_._1)
  val Σs = distributions.map(_._2)

  val outPath = new File("data/", "kf-results.csv")
  Using(new PrintWriter(outPath.toString)) { file =>
    file.println(List("true_x1", "true_x2", "y", "mu1", "mu2", "sigma1", "sigma2").mkString(","))
    for (i <- 0 until T + 1) {
      file.write(s"${List(xTrue(i)(0), xTrue(i)(1), y(i)(0), μs(i)(0), μs(i)(1),
        math.sqrt(Σs(i)(0,0)), math.sqrt(Σs(i)(1,1))).mkString(",")}\n")
    }
  }


}
