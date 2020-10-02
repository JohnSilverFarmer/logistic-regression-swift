//
//  LogisticRegression.swift
//  logistic_regression
//
//  Created by Johannes Silberbauer
//

import Foundation


class LogisticRegression {
    
    var coefs: Matrix
    var intercept: Double
    
    init(nFeatures: Int) {
        self.coefs = .init(rows: nFeatures, cols: 1, repeating: 1)
        self.intercept = 0.5
    }
    
    /// Computes the dot product between input features and regression coefficients.
    func interceptDot(_ x: Matrix) -> Matrix {
        return x * self.coefs + self.intercept
    }
    
    /// Evaluates the log-likelihood for a pair of inputs and outputs.
    func logLikelihood(_ x: Matrix,_ y: Matrix) -> Double {
        precondition(y.cols == 1)
        precondition(x.rows == y.rows)
        let z = interceptDot(x)
        let logl = y.elemMul(matrix: z.sigmoid()) + (-y + 1.0).elemMul(matrix: (-z).sigmoid().log())
        return logl.sum()
    }
    
    /// Computes the gradient of the log-likelihood with respect to the parameters.
    func logLikelihoodGrad(_ x: Matrix, _ y: Matrix) -> (dCoef: Matrix, dIntercept: Double) {
        precondition(y.cols == 1)
        precondition(x.rows == y.rows)
        let scale = y - interceptDot(x).sigmoid()
        let dCoef = scale.diagMul(matrix: x)
        return (dCoef.sumCols().t() / Double(x.rows), scale.sum() / Double(x.rows))
    }
    
    /// Estimate the regression coeficients using the specified input output pairs.
    func fit(_ x: Matrix, _ y: Matrix, steps: Int, alpha: Double) -> [Double] {
        precondition(y.cols == 1)
        precondition(x.rows == y.rows)
        
        var loss: [Double] = .init(repeating: .nan, count: steps)
        
        for step in 0..<steps {
            loss[step] = logLikelihood(x, y)
            let grad = logLikelihoodGrad(x, y)
            
            // maximize likelihood
            coefs = coefs + alpha * grad.dCoef
            intercept = intercept + alpha * grad.dIntercept
        }
        
        return loss
    }
    
    /// Computes P(Y=1|x).
    func predict(_ x: Matrix) -> Matrix {
        return interceptDot(x).sigmoid()
    }
    
    /// Compute loss and accuracy for a pair of inputs and targets.
    func evaluate(_ x: Matrix, _ y: Matrix, thres: Double = 0.5) -> (predictions: Matrix, loss: Double, accuracy: Double) {
        let yPred = predict(x)
        let nCorrect = ((0..<x.rows).map { (abs(y[$0, 0] - 1.0) < 1e-8) == (yPred[$0, 0] > thres) }).filter{$0}.count
        let loss = -logLikelihood(x, y)
        return (yPred, loss, Double(nCorrect) / Double(x.rows))
    }
    
}
