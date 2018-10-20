package com.github.nmasahiro.kf

// (t: Int) => (model matrix: M)
case class StateSpaceModel(
  F: Int => M, // linear factor for x in system model
  G: Int => M, // linear factor for v in system model
  H: Int => M, // linear factor for x in observation model
  Q: Int => M, // covariance matrix that samples v
  R: Int => M // covariance matrix that samples w
)
