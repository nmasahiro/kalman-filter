package com.github.nmasahiro.kf

import breeze.stats.distributions.Rand
import breeze.linalg._
import breeze.linalg.eigSym.EigSym
import scala.collection.immutable.{Vector => SVector}

import scala.annotation.tailrec

object StateAndObs {

  def apply(model: StateSpaceModel, x0: V, T: Int): SVector[(V, V)] = {

    @tailrec
    def run(t: Int, xPrev: V, accum: SVector[(V, V)]): SVector[(V, V)] = {
      if (t == T) {
        accum
      }
      else {
        val EigSym(lambdaQ, evsQ) = eigSym(model.Q(t))
        val sqrtQ = evsQ * diag(lambdaQ.map(q => math.sqrt(q))) * evsQ.t
        val v = DenseVector.zeros[Double](model.G(t).cols).map(_ => Rand.gaussian.draw())
        val x = model.F(t) * xPrev + model.G(t) * (sqrtQ * v)
        val EigSym(lambdaR, evsR) = eigSym(model.R(t))
        val sqrtR = evsR * diag(lambdaR.map(r => math.sqrt(r))) * evsR.t
        val w = DenseVector.zeros[Double](model.H(t).rows).map(_ => Rand.gaussian.draw())
        val y = model.H(t) * x + (sqrtR * w)
        run(t + 1, x, accum :+ ((x, y)))
      }
    }

    val EigSym(lambdaR, evsR) = eigSym(model.R(0))
    val sqrtR = evsR * diag(lambdaR.map(r => math.sqrt(r))) * evsR.t
    val w = DenseVector.zeros[Double](model.H(0).rows).map(_ => Rand.gaussian.draw())
    val y0 = model.H(0) * x0 + (sqrtR * w)
    run(0, x0, SVector[(V, V)]((x0, y0)))
  }

}
