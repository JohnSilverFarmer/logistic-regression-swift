//
//  Matrix.swift
//  logistic_regression
//
//  Created by Johannes Silberbauer
//

import Foundation
import Accelerate


/// A matrix class implementing a basic set of operations. Build on top of the accelerate framework.
/// (Parts of this class where inspired by https://github.com/hollance/Matrix)
class Matrix {
    
    let cols: Int
    let rows: Int
    var data: [Double] // row major storage
    
    init(rows: Int, cols: Int, repeating repeatedValue: Double) {
        self.rows = rows
        self.cols = cols
        self.data = .init(repeating: repeatedValue, count: rows * cols)
    }
    
    init(rows: Int, cols: Int, data: [Double]) {
        precondition(rows * cols == data.count, "Invalid number of elements for matrix dimensions.")
        self.rows = rows
        self.cols = cols
        self.data = data
    }
    
}

// MARK: - Factory Methods

extension Matrix {
    
    static func from(rows: [[Double]]) -> Matrix {
        return Matrix(rows: rows.count, cols: rows[0].count, data: rows.reduce([], +))
    }
    
    static func from(cols: [[Double]]) -> Matrix {
        return Matrix(rows: cols.count, cols: cols[0].count, data: cols.reduce([], +)).t()
    }
    
    static func from(csvFile path: String) throws -> Matrix {
        let csv = try String(contentsOf: URL(fileURLWithPath: path))
        let rows = csv.components(separatedBy: .newlines).filter { !$0.isEmpty }
        let grid = rows.map{
            $0.components(separatedBy: ",").map{
                Double($0) ?? Double.nan
            }
        }
        return Matrix.from(rows: grid)
    }
    
}


// MARK: - Indexing

extension Matrix {
    
    public subscript(row: Int, col: Int) -> Double {
        get {
            return self.data[(row * self.cols) + col]
        }
        set {
            self.data[(row * self.cols) + col] = newValue
        }
    }
    
    public subscript(column c: Int) -> Matrix {
        get {
            let v = Matrix(rows: rows, cols: 1, repeating: .nan)
            self.data.withUnsafeBufferPointer { src in
                v.data.withUnsafeMutableBufferPointer { dst in
                    cblas_dcopy(Int32(rows), src.baseAddress! + c, Int32(cols), dst.baseAddress, 1)
                }
            }
          return v
        }
        set(v) {
            precondition(v.rows == rows && v.cols == 1)
            self.data.withUnsafeMutableBufferPointer { dstPtr in
                v.data.withUnsafeBufferPointer { srcPtr in
                    cblas_dcopy(Int32(rows), srcPtr.baseAddress, 1, dstPtr.baseAddress! + c, Int32(cols))
                }
            }
        }
      }
    
    public subscript(columns range: CountableRange<Int>) -> Matrix {
        get {
            precondition(range.upperBound <= self.cols)
            let result = Matrix(rows: self.rows, cols: range.upperBound - range.lowerBound, repeating: .nan)
            for r in 0..<rows {
                for c in range {
                    result[r, c - range.lowerBound] = self[r, c]
                }
            }
            return result
        }
        set(m) {
            precondition(range.upperBound <= self.cols)
            for r in 0..<self.rows {
                for c in range {
                    self[r, c] = m[r, c - range.lowerBound]
                }
            }
        }
      }
    
}

// MARK: - Transpose

extension Matrix {
    
    /// Return the transpose of the matrix.
    func t() -> Matrix {
        let result = Matrix(rows: self.cols, cols: self.rows, repeating: .nan)
        vDSP_mtransD(self.data, 1, &result.data, 1, vDSP_Length(self.cols), vDSP_Length(self.rows))
        return result
    }
    
}

// MARK: - Summation

extension Matrix {
    
    /// Computes the sum of all elements.
    func sum() -> Double {
        var result: Double = .nan
        vDSP_sveD(self.data, 1, &result, vDSP_Length(self.cols * self.rows))
        return result
    }
    
    /// Compute the sum of each column.
    func sumCols() -> Matrix {
        let result = Matrix(rows: 1, cols: self.cols, repeating: .nan)
        self.data.withUnsafeBufferPointer { selfPtr in
            result.data.withUnsafeMutableBufferPointer { resultPtr in
                for c in 0..<self.cols {
                    vDSP_sveD(selfPtr.baseAddress! + c, self.cols, resultPtr.baseAddress! + c, vDSP_Length(self.rows))
                }
            }
        }
        return result
    }
    
}

// MARK: - Elementwise Functions

extension Matrix {
    
    /// Compute e^x elementwise.
    func exp() -> Matrix {
        return Matrix(rows: self.rows, cols: self.cols, data: vForce.exp(self.data))
    }
    
    /// Calculate the natural logarithm elementwise.
    func log() -> Matrix {
        let result = Matrix(rows: self.rows, cols: self.cols, repeating: .nan)
        var n = Int32(self.data.count)
        vvlog(&result.data, self.data, &n)
        return result
    }
    
    /// Compute 1/x elementwise.
    func reciprocal() -> Matrix {
        return Matrix(rows: self.rows, cols: self.cols, data: vForce.reciprocal(self.data))
    }
    
    /// Compute 1/(1 + e^(-x)) elementwise.
    func sigmoid() -> Matrix {
        return ((-self).exp() + 1.0).reciprocal()
    }
    
}

// MARK: - Operators

extension Matrix {
    
    static func *(lhs: Matrix, rhs: Matrix) -> Matrix {
        precondition(lhs.cols == rhs.rows, "Invalid matrix dimensions for mul.")
        let result = Matrix(rows: lhs.rows, cols: rhs.cols, repeating: .nan)
        vDSP_mmulD(lhs.data, 1, rhs.data, 1, &result.data, 1, vDSP_Length(lhs.rows), vDSP_Length(rhs.cols), vDSP_Length(rhs.rows))
        return result
    }
    
    static func *(lhs: Matrix, rhs: Double) -> Matrix {
        return Matrix(rows: lhs.rows, cols: lhs.cols, data: vDSP.multiply(rhs, lhs.data))
    }
    
    static func *(lhs: Double, rhs: Matrix) -> Matrix {
        return rhs * lhs
    }
    
    static prefix func - (matrix: Matrix) -> Matrix {
            return matrix * (-1)
    }
    
    static func +(lhs: Matrix, rhs: Matrix) -> Matrix {
        precondition(lhs.cols == rhs.cols && lhs.rows == rhs.rows, "Invalid matrix dimensions for add.")
        let result = Matrix(rows: lhs.rows, cols: lhs.cols, repeating: .nan)
        vDSP.add(lhs.data, rhs.data, result: &result.data)
        return result
    }
    
    static func +(lhs: Matrix, rhs: Double) -> Matrix {
        var scalar = rhs
        let result = Matrix(rows: lhs.rows, cols: lhs.cols, repeating: .nan)
        vDSP_vsaddD(lhs.data, 1, &scalar, &result.data, 1, vDSP_Length(lhs.data.count))
        return result
    }
    
    static func -(lhs: Matrix, rhs: Double) -> Matrix {
        return lhs + (-rhs)
    }
    
    static func -(lhs: Matrix, rhs: Matrix) -> Matrix {
        precondition(lhs.cols == rhs.cols && lhs.rows == rhs.rows, "Invalid matrix dimensions for sub.")
        let data = vDSP.subtract(lhs.data, rhs.data)
        return Matrix(rows: rhs.rows, cols: rhs.cols, data: data)
    }
    
    static func /(lhs: Matrix, rhs: Double) -> Matrix {
        let data = vDSP.divide(lhs.data, rhs)
        return Matrix(rows: lhs.rows, cols: lhs.cols, data: data)
    }
    
}

// MARK: - Other Multiplication

extension Matrix {
    
    /// Computes the elementwise product of two matrices.
    func elemMul(matrix: Matrix) -> Matrix {
        let result = Matrix(rows: matrix.rows, cols: matrix.cols, repeating: .nan)
        vDSP_vmulD(self.data, 1, matrix.data, 1, &result.data, 1, vDSP_Length(result.data.count))
        return result
    }
    
    /// Computes the result of diag(self) * A.
    func diagMul(matrix: Matrix) -> Matrix {
        precondition(self.cols == 1 || self.rows == 1, "Cannot convert non-vector to diagonal matrix.")
        precondition(self.cols * self.rows == matrix.rows, "Incompatible dimensions for matrix product.")
        let result = Matrix(rows: matrix.rows, cols: matrix.cols, repeating: .nan)
        for i in 0..<self.data.count {
            var scalar = self.data[i]
            result.data.withUnsafeMutableBufferPointer { resultPtr in
                matrix.data.withUnsafeBufferPointer { matrixPtr in
                    vDSP_vsmulD(matrixPtr.baseAddress! + i * matrix.cols, 1, &scalar, resultPtr.baseAddress! + i * matrix.cols, 1, vDSP_Length(matrix.cols))
                }
            }
        }
        return result
    }
    
}

// MARK: - String Representation

extension Matrix: CustomStringConvertible {
    
    public var description: String {
        var desc = "["
        for row in 0..<self.rows {
            if row != 0 {
                desc += " "
            }
            for col in 0..<self.cols {
                desc += String(self[row, col])
                if col != self.cols - 1 {
                    desc += ", "
                }
            }
            if row != self.rows - 1 {
                desc += "\n"
            }
        }
        desc += "]"
        return desc
    }
    
}
