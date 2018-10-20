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
  val Σ0 = DenseMatrix.eye[Double](xDim) * 0.01
//  val Σ0 = DenseMatrix

  val x0 = DenseVector(math.Pi / 6.0, math.Pi / 12.0)
  // state model
  val F = (_: Int) => DenseMatrix((1.001, -0.01), (0.01, 0.999))
  val estF = (_: Int) => DenseMatrix((1.001, -0.01), (0.01, 0.999))
  val G = (_: Int) => DenseMatrix(0.0001, 0.01) // note that the shape of this matrix is (2, 1)
  val estG = (_: Int) => DenseMatrix(0.0001, 0.01) // note that the shape of this matrix is (2, 1)
  val H = (_: Int) => DenseMatrix(0.0, 1.0).t
  val estH = (_: Int) => DenseMatrix(0.0, 1.0).t

  // noise parameter
  // system noise covariance
  val Q = (_: Int) => DenseMatrix(0.04)
  val estQ = (_: Int) => DenseMatrix(0.04)
  // observation noise covariance
  val R = (_: Int) => DenseMatrix(0.01)
  val estR = (_: Int) => DenseMatrix(0.01)

  val model = StateSpaceModel(F, G, H, Q, R)
  val estimatedModel = StateSpaceModel(estF, estG, estH, estQ, estR)

  val xAndy = StateAndObs(model, x0, T)
  val (xTrue, y) = (xAndy.map(_._1), xAndy.map(_._2))

  val driver = KFDriver(μ0, Σ0, y, estimatedModel,
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
