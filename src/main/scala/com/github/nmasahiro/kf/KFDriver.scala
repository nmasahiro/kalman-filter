package com.github.nmasahiro.kf

import breeze.linalg.inv

import scala.annotation.tailrec
import scala.collection.immutable.{Vector => SVector}

case class KFDriver(μ: V, Σ: M, yVector: SVector[V], model: StateSpaceModel, stopCondition: StopCondition) {

  @tailrec
  final def estimate(t: Int, μ: V, Σ: M, accum: SVector[(V, M)]): SVector[(V, M)] = {
    // stopCondition check
    if (stopCondition((t, Σ))) {
      println(s"stopCondition: t:$t, Σ:$Σ")
      accum
    } else {
      // prediction
      val μPred = model.F(t) * μ
      val ΣPred = model.F(t) * Σ * model.F(t).t + model.G(t) * model.Q(t) * model.G(t).t

      // filtering
      val K = ΣPred * model.H(t).t * inv(model.H(t) * ΣPred * model.H(t).t + model.R(t))
      val μFiltered = μPred + K * (yVector(t) - model.H(t) * μPred)
      val ΣFiltered = ΣPred - K * model.H(t) * ΣPred

      estimate(t + 1, μFiltered, ΣFiltered, accum :+ ((μFiltered, ΣFiltered)))
    }
  }

  def estimate(): SVector[(V, M)] = estimate(0, μ, Σ, SVector((μ, Σ)))

}
