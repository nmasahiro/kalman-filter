package com.github.nmasahiro

import breeze.linalg._
import breeze.linalg.eigSym.EigSym

package object kf {

  type V = DenseVector[Double]

  type M = DenseMatrix[Double]

  type StopCondition = PartialFunction[(Int, M), Boolean]

  val proceed: StopCondition = {
    case (_: Int, _: M) => false
  }

  def iterationsExceed(maxItr: Int): StopCondition = {
    case (itr: Int, _) if itr >= maxItr => true
  }

  def maxEigConverged(maxEig: Double): StopCondition = {
    case (_, cov: M) if max(eigSym(cov).eigenvalues) <= maxEig => true
  }

}
