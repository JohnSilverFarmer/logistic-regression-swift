//
//  main.swift
//  logistic_regression
//
//  Created by Johannes Silberbauer
//

import Foundation


let dataPath = CommandLine.arguments.last!
guard dataPath.hasSuffix(".csv") else {
    preconditionFailure("No input data file found.")
}

guard let data = try? Matrix.from(csvFile: dataPath) else {
    preconditionFailure("Invalid dataset at \(dataPath)")
}

let x = data[columns: 0..<(data.cols - 1)]
let y = data[column: data.cols - 1]
print("\nUsing dataset:", dataPath.components(separatedBy: "/").last!)

let clf = LogisticRegression(nFeatures: x.cols)

// eval initial accuracy
let evalNoTrain = clf.evaluate(x, y)
print("Initial accuracy =", evalNoTrain.accuracy, "| loss =", evalNoTrain.loss, "\n")

// fit the model
let loss = clf.fit(x, y, steps: 2000, alpha: 0.5)
print("Log-Likelihood at last gradient step = ", loss[loss.count - 1])

// eval learned model on train set
let eval = clf.evaluate(x, y)
print("Final accuracy =", eval.accuracy, "| loss =", eval.loss)
print("Learned coeficients =", clf.coefs.data + [clf.intercept], "\n")
