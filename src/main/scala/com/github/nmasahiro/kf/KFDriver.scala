package com.github.nmasahiro.kf

import breeze.linalg.inv

import scala.annotation.tailrec
import scala.collection.immutable.{Vector => SVector}

case class KFDriver(μ: V, Σ: M, yVector: SVector[V], model: StateSpaceModel, stopCondition: StopCondition) {

  @tailrec
  final def estimate(t: Int, μ: V, Σ: M, accum: SVector[(V, M)]): SVector[(V, M)] = {
    // stopCondition check
    if (stopCondition.apply((t, Σ))) {
      println(s"stopCondition: t:$t, Σ:$Σ")
      accum
    } else {
      // prediction
      val μPred = model.F * μ
      val ΣPred = model.F * Σ * model.F.t + model.G * model.Q * model.G.t

      // filtering
      val K = ΣPred * model.H.t * inv(model.H * ΣPred * model.H.t + model.R)
      val μFiltered = μPred + K * (yVector(t) - model.H * μPred)
      val ΣFiltered = ΣPred - K * model.H * ΣPred

      estimate(t + 1, μFiltered, ΣFiltered, accum :+ (μFiltered, ΣFiltered))
    }
  }

  def estimate(): SVector[(V, M)] = estimate(0, μ, Σ, SVector((μ, Σ)))

}
