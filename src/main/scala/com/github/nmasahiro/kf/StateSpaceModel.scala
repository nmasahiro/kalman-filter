package com.github.nmasahiro.kf

import breeze.linalg.DenseMatrix

case class StateSpaceModel(
  F: M, // linear factor for x in system model
  G: M, // linear factor for v in system model
  H: M, // linear factor for x in observation model
  Q: M, // covariance matrix that samples v
  R: M // covariance matrix that samples w
)
