use rand::prelude::*;
use rand_distr::{Normal, Distribution};
use std::rc::Rc;
use std::cell::RefCell;
use std::fmt;

#[derive(Debug, Clone)]
pub struct Matrix {
    pub data: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![0.0; rows * cols],
            rows,
            cols,
        }
    }

    pub fn from_vec(rows: usize, cols: usize, data: Vec<f64>) -> Result<Self, String> {
        if data.len() != rows * cols {
            return Err("Data length does not match dimensions".to_string());
        }
        Ok(Self { data, rows, cols })
    }

    pub fn dot(&self, other: &Matrix) -> Result<Matrix, String> {
        if self.cols != other.rows {
            return Err(format!("Dimensions mismatch: {}x{} and {}x{}", self.rows, self.cols, other.rows, other.cols));
        }
        let mut res = Matrix::new(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.data[i * self.cols + k] * other.data[k * other.cols + j];
                }
                res.data[i * other.cols + j] = sum;
            }
        }
        Ok(res)
    }

    pub fn add(&self, other: &Matrix) -> Result<Matrix, String> {
        if self.rows != other.rows {
            return Err("Row dimension mismatch for addition".to_string());
        }
        
        let mut result = Matrix::new(self.rows, self.cols);
        if self.cols == other.cols {
            for i in 0..self.data.len() {
                result.data[i] = self.data[i] + other.data[i];
            }
        } else if other.cols == 1 {
            // Broadcasting: other is a column vector
            for i in 0..self.rows {
                for j in 0..self.cols {
                    result.data[i * self.cols + j] = self.data[i * self.cols + j] + other.data[i];
                }
            }
        } else {
            return Err("Column dimension mismatch for addition (and not broadcasting)".to_string());
        }
        Ok(result)
    }

    pub fn subtract(&self, other: &Matrix) -> Result<Matrix, String> {
        if self.rows != other.rows {
            return Err("Row dimension mismatch for subtraction".to_string());
        }
        let mut result = Matrix::new(self.rows, self.cols);
        if self.cols == other.cols {
            for i in 0..self.data.len() {
                result.data[i] = self.data[i] - other.data[i];
            }
        } else if other.cols == 1 {
            // Broadcasting
            for i in 0..self.rows {
                for j in 0..self.cols {
                    result.data[i * self.cols + j] = self.data[i * self.cols + j] - other.data[i];
                }
            }
        } else {
            return Err("Column dimension mismatch for subtraction".to_string());
        }
        Ok(result)
    }

    pub fn multiply_elements(&self, other: &Matrix) -> Result<Matrix, String> {
        if self.rows != other.rows {
            return Err("Row dimension mismatch for element-wise multiplication".to_string());
        }
        let mut result = Matrix::new(self.rows, self.cols);
        if self.cols == other.cols {
            for i in 0..self.data.len() {
                result.data[i] = self.data[i] * other.data[i];
            }
        } else if other.cols == 1 {
            // Broadcasting
            for i in 0..self.rows {
                for j in 0..self.cols {
                    result.data[i * self.cols + j] = self.data[i * self.cols + j] * other.data[i];
                }
            }
        } else {
            return Err("Column dimension mismatch for element-wise multiplication".to_string());
        }
        Ok(result)
    }

    pub fn transpose(&self) -> Matrix {
        let mut result = Matrix::new(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[j * self.rows + i] = self.data[i * self.cols + j];
            }
        }
        result
    }

    pub fn apply(&mut self, f: fn(f64) -> f64) {
        for val in self.data.iter_mut() {
            *val = f(*val);
        }
    }

    pub fn map(&self, f: fn(f64) -> f64) -> Matrix {
        let mut result = self.clone();
        for val in result.data.iter_mut() {
            *val = f(*val);
        }
        result
    }

    pub fn reshape(&self, rows: usize, cols: usize) -> Result<Matrix, String> {
        if rows * cols != self.data.len() {
            return Err(format!("Cannot reshape {} elements into {}x{}", self.data.len(), rows, cols));
        }
        let mut res = self.clone();
        res.rows = rows;
        res.cols = cols;
        Ok(res)
    }

    pub fn get_column(&self, col: usize) -> Vec<f64> {
        let mut res = Vec::with_capacity(self.rows);
        for i in 0..self.rows {
            res.push(self.data[i * self.cols + col]);
        }
        res
    }

    pub fn gather_rows(&self, indices: &[usize]) -> Matrix {
        let mut res_data = Vec::with_capacity(indices.len() * self.cols);
        for &idx in indices {
            let start = idx * self.cols;
            let end = start + self.cols;
            res_data.extend_from_slice(&self.data[start..end]);
        }
        Matrix::from_vec(indices.len(), self.cols, res_data).unwrap()
    }

    pub fn apply_causal_mask(&self) -> Matrix {
        let mut res = self.clone();
        for i in 0..self.rows {
            for j in 0..self.cols {
                if j > i {
                    res.data[i * self.cols + j] = -1e15; // Negatif sonsuz (yaklaşık)
                }
            }
        }
        res
    }

    pub fn powi(&self, n: i32) -> Matrix {
        let mut res = self.clone();
        for x in res.data.iter_mut() { *x = x.powi(n); }
        res
    }

    pub fn sqrt(&self) -> Matrix {
        let mut res = self.clone();
        for x in res.data.iter_mut() { *x = x.sqrt(); }
        res
    }

    pub fn mean(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        self.data.iter().sum::<f64>() / self.data.len() as f64
    }

    pub fn variance(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        let m = self.mean();
        self.data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / self.data.len() as f64
    }

    pub fn div_scalar(&self, n: f64) -> Matrix {
        let mut res = self.clone();
        for x in res.data.iter_mut() { *x /= n; }
        res
    }
}

pub struct VariableData {
    pub id: usize,
    pub data: Matrix,
    pub grad: Matrix,
    pub backward: Option<Box<dyn Fn()>>,
    pub deps: Vec<Variable>,
}

#[derive(Clone)]
pub struct Variable(Rc<RefCell<VariableData>>);

impl Variable {
    pub fn id(&self) -> usize { self.0.borrow().id }
}

impl fmt::Debug for Variable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let v = self.0.borrow();
        f.debug_struct("Variable")
            .field("id", &v.id)
            .field("data", &v.data)
            .field("grad", &v.grad)
            .finish()
    }
}

static mut VAR_ID_COUNTER: usize = 0;
fn next_var_id() -> usize {
    unsafe {
        VAR_ID_COUNTER += 1;
        VAR_ID_COUNTER
    }
}

impl Variable {
    pub fn new(matrix: Matrix) -> Self {
        let grad = Matrix::new(matrix.rows, matrix.cols);
        Self(Rc::new(RefCell::new(VariableData {
            id: next_var_id(),
            data: matrix,
            grad,
            backward: None,
            deps: Vec::new(),
        })))
    }

    pub fn div_scalar(&self, n: f64) -> Variable {
        let res_data = self.0.borrow().data.div_scalar(n);
        let res = Variable::new(res_data);
        let v = self.clone();
        let res_v = res.clone();
        let backward = move || {
            let res_grad = res_v.0.borrow().grad.clone();
            let mut v_borrow = v.0.borrow_mut();
            for i in 0..v_borrow.grad.data.len() {
                v_borrow.grad.data[i] += res_grad.data[i] / n;
            }
        };
        {
            let mut res_borrow = res.0.borrow_mut();
            res_borrow.backward = Some(Box::new(backward));
            res_borrow.deps = vec![self.clone()];
        }
        res
    }

    pub fn data(&self) -> Matrix {
        self.0.borrow().data.clone()
    }

    pub fn grad(&self) -> Matrix {
        self.0.borrow().grad.clone()
    }

    pub fn zero_grad(&self) {
        let mut v = self.0.borrow_mut();
        v.grad = Matrix::new(v.data.rows, v.data.cols);
    }

    pub fn backward(&self) {
        let mut visited = std::collections::HashSet::new();
        let mut topo = Vec::new();

        fn build_topo(v: &Variable, visited: &mut std::collections::HashSet<*const VariableData>, topo: &mut Vec<Variable>) {
            let ptr = v.0.as_ptr() as *const VariableData;
            if !visited.contains(&ptr) {
                visited.insert(ptr);
                for dep in &v.0.borrow().deps {
                    build_topo(dep, visited, topo);
                }
                topo.push(v.clone());
            }
        }

        build_topo(self, &mut visited, &mut topo);

        for v in topo.into_iter().rev() {
            let b = v.0.borrow_mut().backward.take();
            if let Some(func) = b {
                func();
                v.0.borrow_mut().backward = Some(func);
            }
        }
    }

    pub fn add(&self, other: &Variable) -> Variable {
        let v1_data = self.0.borrow().data.clone();
        let v2_data = other.0.borrow().data.clone();
        let res_data = v1_data.add(&v2_data).unwrap();
        let res = Variable::new(res_data);
        let v1 = self.clone();
        let v2 = other.clone();
        let res_v = res.clone();
        
        let backward = move || {
            let res_grad = res_v.0.borrow().grad.clone();
            let v1_shape = (v1.0.borrow().data.rows, v1.0.borrow().data.cols);
            let v2_shape = (v2.0.borrow().data.rows, v2.0.borrow().data.cols);

            // V1 Grad
            {
                let mut v1_borrow = v1.0.borrow_mut();
                if v1_shape == (res_grad.rows, res_grad.cols) {
                    for i in 0..v1_borrow.grad.data.len() { v1_borrow.grad.data[i] += res_grad.data[i]; }
                } else if v1_shape.1 == 1 {
                    for i in 0..v1_borrow.grad.rows {
                        for j in 0..res_grad.cols {
                            v1_borrow.grad.data[i] += res_grad.data[i * res_grad.cols + j];
                        }
                    }
                }
            }

            // V2 Grad
            {
                let mut v2_borrow = v2.0.borrow_mut();
                if v2_shape == (res_grad.rows, res_grad.cols) {
                    for i in 0..v2_borrow.grad.data.len() { v2_borrow.grad.data[i] += res_grad.data[i]; }
                } else if v2_shape.1 == 1 {
                    for i in 0..v2_borrow.grad.rows {
                        for j in 0..res_grad.cols {
                            v2_borrow.grad.data[i] += res_grad.data[i * res_grad.cols + j];
                        }
                    }
                }
            }
        };
        {
            let mut res_borrow = res.0.borrow_mut();
            res_borrow.backward = Some(Box::new(backward));
            res_borrow.deps = vec![self.clone(), other.clone()];
        }
        res
    }

    pub fn subtract(&self, other: &Variable) -> Variable {
        let v1_data = self.0.borrow().data.clone();
        let v2_data = other.0.borrow().data.clone();
        let res_data = v1_data.subtract(&v2_data).unwrap();
        let res = Variable::new(res_data);
        let v1 = self.clone();
        let v2 = other.clone();
        let res_v = res.clone();
        
        let backward = move || {
            let res_grad = res_v.0.borrow().grad.clone();
            let v1_shape = (v1.0.borrow().data.rows, v1.0.borrow().data.cols);
            let v2_shape = (v2.0.borrow().data.rows, v2.0.borrow().data.cols);

            // V1 Grad
            {
                let mut v1_borrow = v1.0.borrow_mut();
                if v1_shape == (res_grad.rows, res_grad.cols) {
                    for i in 0..v1_borrow.grad.data.len() { v1_borrow.grad.data[i] += res_grad.data[i]; }
                } else if v1_shape.1 == 1 {
                    for i in 0..v1_borrow.grad.rows {
                        for j in 0..res_grad.cols {
                            v1_borrow.grad.data[i] += res_grad.data[i * res_grad.cols + j];
                        }
                    }
                }
            }

            // V2 Grad
            {
                let mut v2_borrow = v2.0.borrow_mut();
                if v2_shape == (res_grad.rows, res_grad.cols) {
                    for i in 0..v2_borrow.grad.data.len() { v2_borrow.grad.data[i] -= res_grad.data[i]; }
                } else if v2_shape.1 == 1 {
                    for i in 0..v2_borrow.grad.rows {
                        for j in 0..res_grad.cols {
                            v2_borrow.grad.data[i] -= res_grad.data[i * res_grad.cols + j];
                        }
                    }
                }
            }
        };
        {
            let mut res_borrow = res.0.borrow_mut();
            res_borrow.backward = Some(Box::new(backward));
            res_borrow.deps = vec![self.clone(), other.clone()];
        }
        res
    }

    pub fn dot(&self, other: &Variable) -> Variable {
        let res_data = self.0.borrow().data.dot(&other.0.borrow().data).unwrap_or_else(|e| {
            let l_data = self.0.borrow().data.clone();
            let r_data = other.0.borrow().data.clone();
            panic!("Dot product error: {} (LHS: {}x{}, RHS: {}x{})", e, l_data.rows, l_data.cols, r_data.rows, r_data.cols);
        });
        let res = Variable::new(res_data);
        let v1 = self.clone();
        let v2 = other.clone();
        let res_v = res.clone();
        let backward = move || {
            let res_grad = res_v.0.borrow().grad.clone();
            let v1_data = v1.0.borrow().data.clone();
            let v2_data = v2.0.borrow().data.clone();
            let dw1 = res_grad.dot(&v2_data.transpose()).unwrap();
            let mut v1_borrow = v1.0.borrow_mut();
            for i in 0..v1_borrow.grad.data.len() {
                v1_borrow.grad.data[i] += dw1.data[i];
            }
            let dw2 = v1_data.transpose().dot(&res_grad).unwrap();
            let mut v2_borrow = v2.0.borrow_mut();
            for i in 0..v2_borrow.grad.data.len() {
                v2_borrow.grad.data[i] += dw2.data[i];
            }
        };
        {
            let mut res_borrow = res.0.borrow_mut();
            res_borrow.backward = Some(Box::new(backward));
            res_borrow.deps = vec![self.clone(), other.clone()];
        }
        res
    }

    pub fn sigmoid(&self) -> Variable {
        let res_data = self.0.borrow().data.map(|x| 1.0 / (1.0 + (-x).exp()));
        let res = Variable::new(res_data);
        let v = self.clone();
        let res_v = res.clone();
        let backward = move || {
            let res_grad = res_v.0.borrow().grad.clone();
            let res_val = res_v.0.borrow().data.clone();
            let mut v_borrow = v.0.borrow_mut();
            for i in 0..v_borrow.grad.data.len() {
                let s = res_val.data[i];
                v_borrow.grad.data[i] += res_grad.data[i] * s * (1.0 - s);
            }
        };
        {
            let mut res_borrow = res.0.borrow_mut();
            res_borrow.backward = Some(Box::new(backward));
            res_borrow.deps = vec![self.clone()];
        }
        res
    }

    pub fn relu(&self) -> Variable {
        let res_data = self.0.borrow().data.map(|x| if x > 0.0 { x } else { 0.0 });
        let res = Variable::new(res_data);
        let v = self.clone();
        let res_v = res.clone();
        let backward = move || {
            let res_grad = res_v.0.borrow().grad.clone();
            let v_data = v.0.borrow().data.clone();
            let mut v_borrow = v.0.borrow_mut();
            for i in 0..v_borrow.grad.data.len() {
                if v_data.data[i] > 0.0 {
                    v_borrow.grad.data[i] += res_grad.data[i];
                }
            }
        };
        {
            let mut res_borrow = res.0.borrow_mut();
            res_borrow.backward = Some(Box::new(backward));
            res_borrow.deps = vec![self.clone()];
        }
        res
    }

    pub fn mul_elements(&self, other: &Variable) -> Variable {
        let v1_data = self.0.borrow().data.clone();
        let v2_data = other.0.borrow().data.clone();
        let res_data = v1_data.multiply_elements(&v2_data).unwrap();
        let res = Variable::new(res_data);
        let v1 = self.clone();
        let v2 = other.clone();
        let res_v = res.clone();
        
        let backward = move || {
            let res_grad = res_v.0.borrow().grad.clone();
            let v1_data = v1.0.borrow().data.clone();
            let v2_data = v2.0.borrow().data.clone();

            // V1 Grad
            {
                let mut v1_borrow = v1.0.borrow_mut();
                let v1_cols = v1_borrow.grad.cols;
                if v1_data.rows == v2_data.rows && v1_data.cols == v2_data.cols {
                    for i in 0..v1_borrow.grad.data.len() {
                        v1_borrow.grad.data[i] += res_grad.data[i] * v2_data.data[i];
                    }
                } else if v1_data.cols == 1 && v2_data.cols > 1 {
                    for i in 0..v1_borrow.grad.rows {
                        for j in 0..v2_data.cols {
                            v1_borrow.grad.data[i] += res_grad.data[i * v2_data.cols + j] * v2_data.data[i * v2_data.cols + j];
                        }
                    }
                } else if v2_data.cols == 1 && v1_data.cols > 1 {
                    for i in 0..v1_borrow.grad.rows {
                        for j in 0..v1_cols {
                            v1_borrow.grad.data[i * v1_cols + j] += res_grad.data[i * v1_cols + j] * v2_data.data[i];
                        }
                    }
                }
            }

            // V2 Grad
            {
                let mut v2_borrow = v2.0.borrow_mut();
                let v2_cols = v2_borrow.grad.cols;
                if v1_data.rows == v2_data.rows && v1_data.cols == v2_data.cols {
                    for i in 0..v2_borrow.grad.data.len() {
                        v2_borrow.grad.data[i] += res_grad.data[i] * v1_data.data[i];
                    }
                } else if v2_data.cols == 1 && v1_data.cols > 1 {
                    for i in 0..v2_borrow.grad.rows {
                        for j in 0..v1_data.cols {
                            v2_borrow.grad.data[i] += res_grad.data[i * v1_data.cols + j] * v1_data.data[i * v1_data.cols + j];
                        }
                    }
                } else if v1_data.cols == 1 && v2_data.cols > 1 {
                    for i in 0..v2_borrow.grad.rows {
                        for j in 0..v2_cols {
                            v2_borrow.grad.data[i * v2_cols + j] += res_grad.data[i * v2_cols + j] * v1_data.data[i];
                        }
                    }
                }
            }
        };
        {
            let mut res_borrow = res.0.borrow_mut();
            res_borrow.backward = Some(Box::new(backward));
            res_borrow.deps = vec![self.clone(), other.clone()];
        }
        res
    }

    pub fn map(&self, f: fn(f64) -> f64) -> Variable {
        let res_data = self.0.borrow().data.map(f);
        let res = Variable::new(res_data);
        res
    }

    pub fn gather_rows(&self, indices: Vec<usize>) -> Variable {
        let res_data = self.0.borrow().data.gather_rows(&indices);
        let res = Variable::new(res_data);
        let v = self.clone();
        let res_v = res.clone();
        let idx_copy = indices;
        let backward = move || {
            let res_grad = res_v.0.borrow().grad.clone();
            let mut v_borrow = v.0.borrow_mut();
            let cols = v_borrow.grad.cols;
            for (i, &idx) in idx_copy.iter().enumerate() {
                let start = idx * cols;
                let res_start = i * res_grad.cols;
                for j in 0..res_grad.cols {
                    v_borrow.grad.data[start + j] += res_grad.data[res_start + j];
                }
            }
        };
        {
            let mut res_borrow = res.0.borrow_mut();
            res_borrow.backward = Some(Box::new(backward));
            res_borrow.deps = vec![self.clone()];
        }
        res
    }

    pub fn apply_causal_mask(&self) -> Variable {
        let res_data = self.0.borrow().data.apply_causal_mask();
        let res = Variable::new(res_data);
        let v = self.clone();
        let res_v = res.clone();
        let backward = move || {
            let res_grad = res_v.0.borrow().grad.clone();
            let mut v_borrow = v.0.borrow_mut();
            let rows = res_grad.rows;
            let cols = res_grad.cols;
            for i in 0..rows {
                for j in 0..cols {
                    if j <= i {
                        v_borrow.grad.data[i * cols + j] += res_grad.data[i * cols + j];
                    }
                }
            }
        };
        {
            let mut res_borrow = res.0.borrow_mut();
            res_borrow.backward = Some(Box::new(backward));
            res_borrow.deps = vec![self.clone()];
        }
        res
    }

    pub fn transpose(&self) -> Variable {
        let res_data = self.0.borrow().data.transpose();
        let res = Variable::new(res_data);
        let v = self.clone();
        let res_v = res.clone();
        let backward = move || {
            let res_grad = res_v.0.borrow().grad.clone();
            let mut v_borrow = v.0.borrow_mut();
            let grad_transpose = res_grad.transpose();
            for i in 0..v_borrow.grad.data.len() {
                v_borrow.grad.data[i] += grad_transpose.data[i];
            }
        };
        {
            let mut res_borrow = res.0.borrow_mut();
            res_borrow.backward = Some(Box::new(backward));
            res_borrow.deps = vec![self.clone()];
        }
        res
    }
}


pub trait Layer {
    fn forward(&mut self, input: &Variable, training: bool) -> Variable;
    fn parameters(&self) -> Vec<Variable> {
        Vec::new()
    }
}

pub struct DenseLayer {
    pub weights: Variable,
    pub bias: Variable,
    pub activation: Activation,
}

impl DenseLayer {
    pub fn new(input_size: usize, output_size: usize, activation: Activation) -> Self {
        let mut rng = thread_rng();
        let std_dev = match activation {
            Activation::ReLU => (2.0 / input_size as f64).sqrt(),
            Activation::Sigmoid | Activation::Softmax | Activation::None => (1.0 / input_size as f64).sqrt(),
        };
        let dist = Normal::new(0.0, std_dev).unwrap();
        let weights_data: Vec<f64> = (0..output_size * input_size)
            .map(|_| dist.sample(&mut rng))
            .collect();
        let bias_data = vec![0.01; output_size];
        Self {
            weights: Variable::new(Matrix::from_vec(output_size, input_size, weights_data).unwrap()),
            bias: Variable::new(Matrix::from_vec(output_size, 1, bias_data).unwrap()),
            activation,
        }
    }
}

impl Layer for DenseLayer {
    fn forward(&mut self, input: &Variable, _training: bool) -> Variable {
        let z = self.weights.dot(input).add(&self.bias);
        match self.activation {
            Activation::ReLU => z.relu(),
            Activation::Sigmoid => z.sigmoid(),
            Activation::Softmax => z.softmax(),
            Activation::None => z,
        }
    }
    fn parameters(&self) -> Vec<Variable> {
        vec![self.weights.clone(), self.bias.clone()]
    }
}

pub struct Dropout {
    pub rate: f64,
    pub mask: Option<Variable>,
}

impl Dropout {
    pub fn new(rate: f64) -> Self {
        Self { rate, mask: None }
    }
}

impl Layer for Dropout {
    fn forward(&mut self, input: &Variable, training: bool) -> Variable {
        if !training {
            return input.clone();
        }
        let mut rng = thread_rng();
        let rows = input.0.borrow().data.rows;
        let cols = input.0.borrow().data.cols;
        let mask_data = Matrix::from_vec(
            rows,
            cols,
            (0..rows * cols)
                .map(|_| if rng.gen::<f64>() > self.rate { 1.0 / (1.0 - self.rate) } else { 0.0 })
                .collect()
        ).unwrap();
        let mask_v = Variable::new(mask_data);
        self.mask = Some(mask_v.clone());
        input.mul_elements(&mask_v)
    }
}

pub struct Conv2D {
    pub filters: Variable,
    pub bias: Variable,
    pub stride: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
}

impl Conv2D {
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize) -> Self {
        let mut rng = thread_rng();
        let std_dev = (2.0 / (in_channels * kernel_size * kernel_size) as f64).sqrt();
        let dist = Normal::new(0.0, std_dev).unwrap();
        let filters_data: Vec<f64> = (0..out_channels * in_channels * kernel_size * kernel_size)
            .map(|_| dist.sample(&mut rng))
            .collect();
        let bias_data = vec![0.01; out_channels];
        Self {
            filters: Variable::new(Matrix::from_vec(out_channels, in_channels * kernel_size * kernel_size, filters_data).unwrap()),
            bias: Variable::new(Matrix::from_vec(out_channels, 1, bias_data).unwrap()),
            stride: 1,
            in_channels,
            out_channels,
            kernel_size,
        }
    }
}

impl Layer for Conv2D {
    fn forward(&mut self, input: &Variable, _training: bool) -> Variable {
        let in_data = input.data();
        let in_h = (in_data.data.len() as f64).sqrt() as usize; 
        let in_w = in_h;
        let out_h = (in_h - self.kernel_size) / self.stride + 1;
        let out_w = (in_w - self.kernel_size) / self.stride + 1;
        let mut columns = Vec::new();
        for y in 0..out_h {
            for x in 0..out_w {
                for ky in 0..self.kernel_size {
                    for kx in 0..self.kernel_size {
                        let iy = y * self.stride + ky;
                        let ix = x * self.stride + kx;
                        columns.push(in_data.data[iy * in_w + ix]);
                    }
                }
            }
        }
        let col_matrix = Variable::new(Matrix::from_vec(out_h * out_w, self.kernel_size * self.kernel_size, columns).unwrap());
        let conv_out = self.filters.dot(&col_matrix.transpose());
        conv_out.add(&self.bias)
    }
    fn parameters(&self) -> Vec<Variable> {
        vec![self.filters.clone(), self.bias.clone()]
    }
}

pub struct Flatten;

impl Layer for Flatten {
    fn forward(&mut self, input: &Variable, _training: bool) -> Variable {
        let data = input.data();
        Variable::new(Matrix::from_vec(data.data.len(), 1, data.data.clone()).unwrap())
    }
}

pub trait Optimizer {
    fn step(&mut self, params: &mut [Variable]);
}

pub struct SGD {
    pub lr: f64,
}

impl Optimizer for SGD {
    fn step(&mut self, params: &mut [Variable]) {
        for p in params {
            let mut v = p.0.borrow_mut();
            for i in 0..v.data.data.len() {
                v.data.data[i] -= self.lr * v.grad.data[i];
            }
        }
    }
}

pub struct Adam {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub t: usize,
    pub m: std::collections::HashMap<usize, Matrix>,
    pub v: std::collections::HashMap<usize, Matrix>,
}

impl Adam {
    pub fn new(lr: f64) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            t: 0,
            m: std::collections::HashMap::new(),
            v: std::collections::HashMap::new(),
        }
    }
}

impl Optimizer for Adam {
    fn step(&mut self, params: &mut [Variable]) {
        self.t += 1;
        let t = self.t as i32;
        for p in params {
            let id = p.id();
            let mut var = p.0.borrow_mut();
            
            // Borrow checker hatasını önlemek için gradyan verilerini kopyalıyoruz
            let g_data = var.grad.data.clone();
            let rows = var.grad.rows;
            let cols = var.grad.cols;

            let m = self.m.entry(id).or_insert(Matrix::new(rows, cols));
            let v = self.v.entry(id).or_insert(Matrix::new(rows, cols));

            for i in 0..g_data.len() {
                m.data[i] = self.beta1 * m.data[i] + (1.0 - self.beta1) * g_data[i];
                v.data[i] = self.beta2 * v.data[i] + (1.0 - self.beta2) * g_data[i].powi(2);

                let m_hat = m.data[i] / (1.0 - self.beta1.powi(t));
                let v_hat = v.data[i] / (1.0 - self.beta2.powi(t));

                var.data.data[i] -= self.lr * m_hat / (v_hat.sqrt() + self.epsilon);
            }
        }
    }
}

pub struct LayerNorm {
    pub gamma: Variable,
    pub beta: Variable,
    pub epsilon: f64,
}

impl LayerNorm {
    pub fn new(size: usize) -> Self {
        Self {
            gamma: Variable::new(Matrix::from_vec(size, 1, vec![1.0; size]).unwrap()),
            beta: Variable::new(Matrix::from_vec(size, 1, vec![0.0; size]).unwrap()),
            epsilon: 1e-5,
        }
    }
}

impl Layer for LayerNorm {
    fn forward(&mut self, input: &Variable, _training: bool) -> Variable {
        let data = input.data();
        let mean = data.mean();
        let var = data.variance();
        let std = (var + self.epsilon).sqrt();
        
        // (x - mean) / std
        let mut norm_data = data.clone();
        for x in norm_data.data.iter_mut() {
            *x = (*x - mean) / std;
        }

        let res_data = norm_data.multiply_elements(&self.gamma.data()).unwrap().add(&self.beta.data()).unwrap();
        let res = Variable::new(res_data);
        
        let v_in = input.clone();
        let v_gamma = self.gamma.clone();
        let v_beta = self.beta.clone();
        let res_v = res.clone();
        let eps = self.epsilon;

        let backward = move || {
            let res_grad = res_v.0.borrow().grad.clone();
            let v_g = v_gamma.clone();
            let v_i = v_in.clone();
            let g_data = v_g.0.borrow().data.clone();
            let in_data = v_i.0.borrow().data.clone();
            
            let mean = in_data.mean();
            let var = in_data.variance();
            let std = (var + eps).sqrt();
            let rows = in_data.rows;
            let cols = in_data.cols;

            // Beta grad
            {
                let mut b_borrow = v_beta.0.borrow_mut();
                for i in 0..rows {
                    for j in 0..cols {
                        b_borrow.grad.data[i] += res_grad.data[i * cols + j];
                    }
                }
            }

            // Gamma grad
            {
                let mut g_borrow = v_gamma.0.borrow_mut();
                for i in 0..rows {
                    for j in 0..cols {
                        let norm_val = (in_data.data[i * cols + j] - mean) / std;
                        g_borrow.grad.data[i] += res_grad.data[i * cols + j] * norm_val;
                    }
                }
            }

            // Input grad (simplified)
            {
                let mut in_borrow = v_in.0.borrow_mut();
                for i in 0..in_borrow.grad.data.len() {
                    let r = i / cols;
                    in_borrow.grad.data[i] += res_grad.data[i] * g_data.data[r] / std;
                }
            }
        };

        {
            let mut res_borrow = res.0.borrow_mut();
            res_borrow.backward = Some(Box::new(backward));
            res_borrow.deps = vec![input.clone(), self.gamma.clone(), self.beta.clone()];
        }
        res
    }

    fn parameters(&self) -> Vec<Variable> {
        vec![self.gamma.clone(), self.beta.clone()]
    }
}

impl Matrix {
    pub fn max(&self) -> f64 {
        self.data.iter().copied().fold(f64::NEG_INFINITY, f64::max)
    }

    pub fn argmax(&self) -> usize {
        self.data.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(index, _)| index)
            .unwrap_or(0)
    }
}

pub struct MaxPooling {
    pub pool_size: usize,
    pub stride: usize,
}

impl MaxPooling {
    pub fn new(pool_size: usize, stride: usize) -> Self {
        Self { pool_size, stride }
    }
}

impl Layer for MaxPooling {
    fn forward(&mut self, input: &Variable, _training: bool) -> Variable {
        let in_data = input.data();
        let in_h = (in_data.data.len() as f64).sqrt() as usize;
        let in_w = in_h;
        let out_h = (in_h - self.pool_size) / self.stride + 1;
        let out_w = (in_w - self.pool_size) / self.stride + 1;

        let mut pooled_data = Vec::new();
        let mut max_indices = Vec::new();
        
        for y in 0..out_h {
            for x in 0..out_w {
                let mut max_val = f64::NEG_INFINITY;
                let mut max_idx = 0;
                for py in 0..self.pool_size {
                    for px in 0..self.pool_size {
                        let iy = y * self.stride + py;
                        let ix = x * self.stride + px;
                        let idx = iy * in_w + ix;
                        let val = in_data.data[idx];
                        if val > max_val { 
                            max_val = val; 
                            max_idx = idx;
                        }
                    }
                }
                pooled_data.push(max_val);
                max_indices.push(max_idx);
            }
        }

        let res_data = Matrix::from_vec(out_h * out_w, 1, pooled_data).unwrap();
        let res = Variable::new(res_data);
        
        let v = input.clone();
        let res_v = res.clone();
        let indices = max_indices;
        
        let backward = move || {
            let res_grad = res_v.0.borrow().grad.clone();
            let mut v_borrow = v.0.borrow_mut();
            
            for (i, &idx) in indices.iter().enumerate() {
                v_borrow.grad.data[idx] += res_grad.data[i];
            }
        };
        
        {
            let mut res_borrow = res.0.borrow_mut();
            res_borrow.backward = Some(Box::new(backward));
            res_borrow.deps = vec![input.clone()];
        }
        res
    }
}

impl Variable {
    pub fn softmax(&self) -> Variable {
        let data = self.data();
        let rows = data.rows;
        let cols = data.cols;
        let mut res_data = Matrix::new(rows, cols);
        
        // Auto-axis: If it's a column vector, softmax over rows. Otherwise, softmax over columns (row-wise).
        if cols == 1 {
            let mut max_val = f64::NEG_INFINITY;
            for i in 0..rows {
                if data.data[i] > max_val { max_val = data.data[i]; }
            }
            let mut sum_exp = 0.0;
            for i in 0..rows {
                let val = (data.data[i] - max_val).exp();
                res_data.data[i] = val;
                sum_exp += val;
            }
            for i in 0..rows {
                res_data.data[i] /= sum_exp;
            }
        } else {
            for i in 0..rows {
                let mut max_val = f64::NEG_INFINITY;
                for j in 0..cols {
                    let val = data.data[i * cols + j];
                    if val > max_val { max_val = val; }
                }
                let mut sum_exp = 0.0;
                for j in 0..cols {
                    let val = (data.data[i * cols + j] - max_val).exp();
                    res_data.data[i * cols + j] = val;
                    sum_exp += val;
                }
                for j in 0..cols {
                    res_data.data[i * cols + j] /= sum_exp;
                }
            }
        }
        
        let res = Variable::new(res_data);
        let v = self.clone();
        let res_v = res.clone();
        let backward = move || {
            let res_grad = res_v.0.borrow().grad.clone();
            let res_val = res_v.0.borrow().data.clone();
            let mut v_borrow = v.0.borrow_mut();
            let rows = res_grad.rows;
            let cols = res_grad.cols;
            
            if cols == 1 {
                let mut dot_product = 0.0;
                for i in 0..rows {
                    dot_product += res_val.data[i] * res_grad.data[i];
                }
                for i in 0..rows {
                    let s = res_val.data[i];
                    v_borrow.grad.data[i] += s * (res_grad.data[i] - dot_product);
                }
            } else {
                for i in 0..rows {
                    let mut dot_product = 0.0;
                    for j in 0..cols {
                        dot_product += res_val.data[i * cols + j] * res_grad.data[i * cols + j];
                    }
                    for j in 0..cols {
                        let s = res_val.data[i * cols + j];
                        v_borrow.grad.data[i * cols + j] += s * (res_grad.data[i * cols + j] - dot_product);
                    }
                }
            }
        };
        {
            let mut res_borrow = res.0.borrow_mut();
            res_borrow.backward = Some(Box::new(backward));
            res_borrow.deps = vec![self.clone()];
        }
        res
    }
}

pub enum Activation {
    ReLU,
    Sigmoid,
    Softmax,
    None,
}
pub struct Residual {
    pub layer: Box<dyn Layer>,
}

impl Layer for Residual {
    fn forward(&mut self, input: &Variable, training: bool) -> Variable {
        let output = self.layer.forward(input, training);
        output.add(input)
    }

    fn parameters(&self) -> Vec<Variable> {
        self.layer.parameters()
    }
}

pub struct MultiHeadAttention {
    pub h: usize,
    pub d_k: usize,
    pub w_q: DenseLayer,
    pub w_k: DenseLayer,
    pub w_v: DenseLayer,
    pub w_o: DenseLayer,
    pub causal: bool,
}

impl MultiHeadAttention {
    pub fn new(d_model: usize, h: usize, causal: bool) -> Self {
        let d_k = d_model / h;
        Self {
            h,
            d_k,
            w_q: DenseLayer::new(d_model, d_model, Activation::None), 
            w_k: DenseLayer::new(d_model, d_model, Activation::None),
            w_v: DenseLayer::new(d_model, d_model, Activation::None),
            w_o: DenseLayer::new(d_model, d_model, Activation::None),
            causal,
        }
    }

    fn attention(q: &Variable, k: &Variable, v: &Variable, d_k: usize, causal: bool) -> Variable {
        // q, k, v: [d_model, seq_len]
        
        // 1. scores = Q^T * K / sqrt(d_k) -> [seq_len, seq_len]
        let mut scores = q.transpose().dot(k).div_scalar((d_k as f64).sqrt());
        
        if causal {
            scores = scores.apply_causal_mask();
        }
        
        // 2. weights = softmax(scores) -> [seq_len, seq_len] (her satır toplamı 1)
        let weights = scores.softmax();
        
        // 3. Attention Out = (weights * V^T)^T -> [d_model, seq_len]
        // Bu yapı "Softmax(QK^T)V" formülünün d_model x seq_len konvansiyonumuza uyarlanmış halidir.
        weights.dot(&v.transpose()).transpose()
    }
}

impl Layer for MultiHeadAttention {
    fn forward(&mut self, input: &Variable, training: bool) -> Variable {
        let q = self.w_q.forward(input, training);
        let k = self.w_k.forward(input, training);
        let v = self.w_v.forward(input, training);
        let attn = Self::attention(&q, &k, &v, self.d_k, self.causal);
        self.w_o.forward(&attn, training)
    }

    fn parameters(&self) -> Vec<Variable> {
        let mut params = Vec::new();
        params.extend(self.w_q.parameters());
        params.extend(self.w_k.parameters());
        params.extend(self.w_v.parameters());
        params.extend(self.w_o.parameters());
        params
    }
}

pub struct Reshape {
    pub rows: usize,
    pub cols: usize,
}

impl Reshape {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self { rows, cols }
    }
}

impl Layer for Reshape {
    fn forward(&mut self, input: &Variable, _training: bool) -> Variable {
        Variable::new(input.data().reshape(self.rows, self.cols).unwrap())
    }
}

pub struct FeedForward {
    pub dense1: DenseLayer,
    pub dense2: DenseLayer,
}

impl FeedForward {
    pub fn new(d_model: usize, d_ff: usize) -> Self {
        Self {
            dense1: DenseLayer::new(d_model, d_ff, Activation::ReLU),
            dense2: DenseLayer::new(d_ff, d_model, Activation::ReLU), // ReLU sadece yer tutucu
        }
    }
}

impl Layer for FeedForward {
    fn forward(&mut self, input: &Variable, training: bool) -> Variable {
        let x = self.dense1.forward(input, training);
        self.dense2.forward(&x, training)
    }

    fn parameters(&self) -> Vec<Variable> {
        let mut params = Vec::new();
        params.extend(self.dense1.parameters());
        params.extend(self.dense2.parameters());
        params
    }
}

pub struct TransformerBlock {
    pub attention: MultiHeadAttention,
    pub norm1: LayerNorm,
    pub ffn: FeedForward,
    pub norm2: LayerNorm,
    pub dropout: Dropout,
}

impl TransformerBlock {
    pub fn new(d_model: usize, h: usize, d_ff: usize, dropout_rate: f64, causal: bool) -> Self {
        Self {
            attention: MultiHeadAttention::new(d_model, h, causal),
            norm1: LayerNorm::new(d_model),
            ffn: FeedForward::new(d_model, d_ff),
            norm2: LayerNorm::new(d_model),
            dropout: Dropout::new(dropout_rate),
        }
    }
}

impl Layer for TransformerBlock {
    fn forward(&mut self, input: &Variable, training: bool) -> Variable {
        // Sublayer 1: Attention + Residual + LayerNorm
        let attn_out = self.attention.forward(input, training);
        let x = self.norm1.forward(&attn_out.add(input), training);

        // Sublayer 2: FFN + Residual + LayerNorm
        let ffn_out = self.ffn.forward(&x, training);
        let res = self.norm2.forward(&ffn_out.add(&x), training);

        self.dropout.forward(&res, training)
    }

    fn parameters(&self) -> Vec<Variable> {
        let mut params = Vec::new();
        params.extend(self.attention.parameters());
        params.extend(self.norm1.parameters());
        params.extend(self.ffn.parameters());
        params.extend(self.norm2.parameters());
        params
    }
}

pub struct CharTokenizer {
    pub char_to_id: std::collections::HashMap<char, usize>,
    pub id_to_char: std::collections::HashMap<usize, char>,
    pub vocab_size: usize,
}

impl CharTokenizer {
    pub fn new(text: &str) -> Self {
        let mut chars: Vec<char> = text.chars().collect();
        chars.sort();
        chars.dedup();
        let mut char_to_id = std::collections::HashMap::new();
        let mut id_to_char = std::collections::HashMap::new();
        for (i, &c) in chars.iter().enumerate() {
            char_to_id.insert(c, i);
            id_to_char.insert(i, c);
        }
        let vocab_size = chars.len();
        Self { char_to_id, id_to_char, vocab_size }
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.chars().map(|c| *self.char_to_id.get(&c).unwrap_or(&0)).collect()
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter().map(|id| *self.id_to_char.get(id).unwrap_or(&' ')).collect()
    }
}

pub struct Embedding {
    pub weights: Variable,
}

impl Embedding {
    pub fn new(vocab_size: usize, d_model: usize) -> Self {
        let mut rng = thread_rng();
        let dist = Normal::new(0.0, 0.02).unwrap();
        let data: Vec<f64> = (0..vocab_size * d_model).map(|_| dist.sample(&mut rng)).collect();
        Self {
            weights: Variable::new(Matrix::from_vec(vocab_size, d_model, data).unwrap()),
        }
    }
}

impl Layer for Embedding {
    fn forward(&mut self, input: &Variable, _training: bool) -> Variable {
        let indices: Vec<usize> = input.data().data.iter().map(|&x| x as usize).collect();
        // [vocab_size, d_model] -> [seq_len, d_model] -> transpose -> [d_model, seq_len]
        self.weights.gather_rows(indices).transpose()
    }

    fn parameters(&self) -> Vec<Variable> {
        vec![self.weights.clone()]
    }
}

pub struct PositionalEncoding {
    pub weights: Variable,
}

impl PositionalEncoding {
    pub fn new(max_seq_len: usize, d_model: usize) -> Self {
        let mut rng = thread_rng();
        let dist = Normal::new(0.0, 0.02).unwrap();
        let data: Vec<f64> = (0..max_seq_len * d_model).map(|_| dist.sample(&mut rng)).collect();
        Self {
            weights: Variable::new(Matrix::from_vec(max_seq_len, d_model, data).unwrap()),
        }
    }
}

impl Layer for PositionalEncoding {
    fn forward(&mut self, input: &Variable, _training: bool) -> Variable {
        let seq_len = input.data().cols;  // input: [d_model, seq_len]
        let indices: Vec<usize> = (0..seq_len).collect();
        let pe_subset = self.weights.gather_rows(indices).transpose();  // [seq_len, d_model] -> [d_model, seq_len]
        input.add(&pe_subset)
    }

    fn parameters(&self) -> Vec<Variable> {
        vec![self.weights.clone()]
    }
}

pub struct TinyShakespeareGPT {
    pub tokenizer: CharTokenizer,
    pub embedding: Embedding,
    pub pos_encoding: PositionalEncoding,
    pub blocks: Vec<TransformerBlock>,
    pub ln_final: LayerNorm,
    pub lm_head: DenseLayer,
}

impl TinyShakespeareGPT {
    pub fn new(vocab_size: usize, d_model: usize, n_heads: usize, d_ff: usize, n_layers: usize, max_seq_len: usize, tokenizer: CharTokenizer) -> Self {
        let mut blocks = Vec::new();
        for _ in 0..n_layers {
            blocks.push(TransformerBlock::new(d_model, n_heads, d_ff, 0.1, true));
        }
        Self {
            tokenizer,
            embedding: Embedding::new(vocab_size, d_model),
            pos_encoding: PositionalEncoding::new(max_seq_len, d_model),
            blocks,
            ln_final: LayerNorm::new(d_model),
            lm_head: DenseLayer::new(d_model, vocab_size, Activation::None),
        }
    }

    pub fn forward(&mut self, input_ids: &Variable, training: bool) -> Variable {
        let mut x = self.embedding.forward(input_ids, training);
        x = self.pos_encoding.forward(&x, training);
        for block in &mut self.blocks {
            x = block.forward(&x, training);
        }
        x = self.ln_final.forward(&x, training);
        self.lm_head.forward(&x, training)
    }

    pub fn parameters(&self) -> Vec<Variable> {
        let mut params = Vec::new();
        params.extend(self.embedding.parameters());
        params.extend(self.pos_encoding.parameters());
        for block in &self.blocks {
            params.extend(block.parameters());
        }
        params.extend(self.ln_final.parameters());
        params.extend(self.lm_head.parameters());
        params
    }

    pub fn generate(&mut self, prompt: &str, max_new_tokens: usize, temperature: f64) -> String {
        let mut input_ids = self.tokenizer.encode(prompt);
        let mut rng = thread_rng();
        let max_len = self.pos_encoding.weights.data().rows;

        for _ in 0..max_new_tokens {
            let start = if input_ids.len() > max_len { input_ids.len() - max_len } else { 0 };
            let current_input = &input_ids[start..];
            let input_var = Variable::new(Matrix::from_vec(current_input.len(), 1, current_input.iter().map(|&x| x as f64).collect()).unwrap());
            
            let logits = self.forward(&input_var, false);
            let last_token_logits = logits.data().get_column(logits.data().cols - 1);
            
            let mut probs = last_token_logits.iter().map(|x| (x / temperature).exp()).collect::<Vec<f64>>();
            let sum: f64 = probs.iter().sum();
            if sum == 0.0 || sum.is_nan() {
                let max_idx = last_token_logits.iter().enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i).unwrap_or(0);
                input_ids.push(max_idx);
                continue;
            }
            for p in probs.iter_mut() { *p /= sum; }
            
            let mut cumulative = 0.0;
            let r = rng.gen::<f64>();
            let mut next_id = 0;
            for (id, &p) in probs.iter().enumerate() {
                cumulative += p;
                if r <= cumulative {
                    next_id = id;
                    break;
                }
                next_id = id;
            }
            input_ids.push(next_id);
        }
        self.tokenizer.decode(&input_ids)
    }

    pub fn train_on_text(&mut self, text: &str, steps: usize, seq_len: usize, optimizer: &mut dyn Optimizer) {
        let tokens = self.tokenizer.encode(text);
        let n = tokens.len();
        if n <= seq_len { return; }
        let mut rng = thread_rng();

        println!("\n🎭 Shakespeare Egitimi Basliyor...");
        println!("-------------------------------------------");

        for step in 0..steps {
            let start_idx = rng.gen_range(0..n - seq_len - 1);
            let input_ids = &tokens[start_idx..start_idx + seq_len];
            let target_ids = &tokens[start_idx + 1..start_idx + seq_len + 1];

            let input_var = Variable::new(Matrix::from_vec(seq_len, 1, input_ids.iter().map(|&x| x as f64).collect()).unwrap());
            
            let logits = self.forward(&input_var, true);
            let logits_data = logits.data();
            let mut total_loss = 0.0;
            let mut logits_grad = Matrix::new(logits_data.rows, logits_data.cols);
            
            for t in 0..seq_len {
                let col = logits_data.get_column(t);
                let max_l = col.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let exps: Vec<f64> = col.iter().map(|x| (x - max_l).exp()).collect();
                let sum_exps: f64 = exps.iter().sum();
                let probs: Vec<f64> = exps.iter().map(|x| x / sum_exps).collect();
                
                let target = target_ids[t];
                total_loss -= probs[target].ln().max(-50.0);
                
                for i in 0..probs.len() {
                    let grad = if i == target { probs[i] - 1.0 } else { probs[i] };
                    logits_grad.data[i * seq_len + t] = grad / seq_len as f64;
                }
            }

            // Backward manually set grad
            {
                let mut l_borrow = logits.0.borrow_mut();
                l_borrow.grad = logits_grad;
            }
            logits.backward();
            
            let mut params = self.parameters();
            optimizer.step(&mut params);
            for p in &params { p.zero_grad(); }

            if step % 50 == 0 {
                let bar_len = 20;
                let filled = (step as f32 / steps as f32 * bar_len as f32) as usize;
                let not_filled = bar_len - filled;
                let bar: String = (0..filled).map(|_| "■").collect::<String>() + 
                                  &(0..not_filled).map(|_| " ").collect::<String>();
                print!("\rAdim {:>4} |[{}]| Loss: {:.6}", step, bar, total_loss / seq_len as f64);
                use std::io::Write;
                std::io::stdout().flush().unwrap();
            }
        }
        println!("\n✅ Egitim Tamamlandi.");
    }
}

pub struct NeuralNetwork {
    pub layers: Vec<Box<dyn Layer>>,
}

impl NeuralNetwork {
    pub fn new(layers: Vec<Box<dyn Layer>>) -> Self {
        Self { layers }
    }
    pub fn parameters(&self) -> Vec<Variable> {
        let mut params = Vec::new();
        for layer in &self.layers {
            params.extend(layer.parameters());
        }
        params
    }
    pub fn forward(&mut self, input: &Variable, training: bool) -> Variable {
        let mut current = input.clone();
        for layer in &mut self.layers {
            current = layer.forward(&current, training);
        }
        current
    }
    pub fn predict(&mut self, input_vec: &Vec<f64>) -> Vec<f64> {
        let input = Variable::new(Matrix::from_vec(input_vec.len(), 1, input_vec.clone()).unwrap());
        let output = self.forward(&input, false);
        output.data().data
    }
    pub fn train<O: Optimizer>(&mut self, data: &[(Vec<f64>, Vec<f64>)], epochs: usize, batch_size: usize, optimizer: &mut O) -> Vec<f64> {
        let mut rng = thread_rng();
        let mut loss_history = Vec::new();
        println!("\n🚀 Eğitim Başlatılıyor...");
        println!("-------------------------------------------");
        
        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            let mut shuffled_data = data.to_vec();
            shuffled_data.shuffle(&mut rng);
            
            for batch in shuffled_data.chunks(batch_size) {
                let mut params = self.parameters();
                for p in &params { p.zero_grad(); }
                let mut batch_loss = 0.0;
                for (input_vec, target_vec) in batch {
                    let input = Variable::new(Matrix::from_vec(input_vec.len(), 1, input_vec.clone()).unwrap());
                    let target = Matrix::from_vec(target_vec.len(), 1, target_vec.clone()).unwrap();
                    let output = self.forward(&input, true);
                    let diff = output.data().subtract(&target).unwrap();
                    let mut sum_sq = 0.0;
                    for val in &diff.data { sum_sq += val * val; }
                    batch_loss += sum_sq / 2.0;
                    
                    let mut output_grad = output.grad();
                    for i in 0..output_grad.data.len() { output_grad.data[i] = diff.data[i]; }
                    {
                        let mut o_borrow = output.0.borrow_mut();
                        o_borrow.grad = output_grad;
                    }
                    output.backward();
                }
                optimizer.step(&mut params);
                total_loss += batch_loss / batch.len() as f64;
            }
            
            let epoch_loss = total_loss / (data.len() / batch_size) as f64;
            loss_history.push(epoch_loss);

            if epoch % (epochs / 10).max(1) == 0 || epoch == epochs - 1 {
                let progress = (epoch as f64 / epochs as f64 * 20.0) as usize;
                let bar: String = (0..20).map(|i| if i < progress { "■" } else { " " }).collect();
                println!("Epoch {:>4} |[{}]| Loss: {:.6}", epoch, bar, epoch_loss);
            }
        }
        println!("-------------------------------------------");
        println!("✅ Eğitim Tamamlandı.\n");
        loss_history
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_modern_components() {
        let mut model = NeuralNetwork::new(vec![
            Box::new(DenseLayer::new(2, 4, Activation::ReLU)),
            Box::new(LayerNorm::new(4)),
            Box::new(Residual {
                layer: Box::new(DenseLayer::new(4, 4, Activation::ReLU)),
            }),
            Box::new(DenseLayer::new(4, 1, Activation::Sigmoid)),
        ]);
        let xor_data = vec![
            (vec![0.0, 0.0], vec![0.0]),
            (vec![0.0, 1.0], vec![1.0]),
            (vec![1.0, 0.0], vec![1.0]),
            (vec![1.0, 1.0], vec![0.0]),
        ];
        
        let mut optimizer = Adam::new(0.01);
        model.train(&xor_data, 2000, 4, &mut optimizer);
        
        println!("\n--- XOR Modern Bileşen Sonuçları ---");
        for (input_vec, target_vec) in &xor_data {
            let predict = model.predict(input_vec);
            println!("Girdi: {:?}, Hedef: {:?}, Tahmin: {:.4}", input_vec, target_vec, predict[0]);
            if target_vec[0] > 0.5 { assert!(predict[0] > 0.7); }
            else { assert!(predict[0] < 0.3); }
        }
    }

    #[test]
    fn test_mnist_cnn() {
        // 28x28 mock görüntü
        let mut img_data = vec![0.0; 28 * 28];
        img_data[14 * 28 + 14] = 1.0; // Ortada bir nokta
        
        let mut model = NeuralNetwork::new(vec![
            Box::new(Conv2D::new(1, 4, 3)), // 28x28 -> 26x26
            Box::new(MaxPooling::new(2, 2)), // 26x26 -> 13x13
            Box::new(Flatten),               // 13x13x4 -> 676
            Box::new(DenseLayer::new(13 * 13 * 4, 10, Activation::Softmax)),
        ]);

        let dummy_data = vec![
            (img_data, vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]), // Label 5
        ];

        let mut optimizer = Adam::new(0.01);
        model.train(&dummy_data, 5, 1, &mut optimizer);

        let predict = model.predict(&dummy_data[0].0);
        assert_eq!(predict.len(), 10);
        let sum: f64 = predict.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_attention() {
        let d_model = 8;
        let h = 2;
        let seq_len = 5;
        
        let mut mha = MultiHeadAttention::new(d_model, h, false);
        
        // Girdi: (d_model, seq_len) -> (8, 5)
        let input_data = vec![0.5; d_model * seq_len];
        let input = Variable::new(Matrix::from_vec(d_model, seq_len, input_data).unwrap());
        
        let output = mha.forward(&input, false);
        
        // Çıktı boyutu: d_model x seq_len (V matrisi boyutu dk x seq_len ise çıktı dk x seq_len olur)
        // Bizim basitleştirilmiş implementasyonumuzda d_model x seq_len bekliyoruz
        assert_eq!(output.data().rows, d_model);
        assert_eq!(output.data().cols, seq_len);
    }

    #[test]
    fn test_transformer_block() {
        let d_model = 8;
        let h = 2;
        let d_ff = 16;
        let seq_len = 4;
        
        let mut model = NeuralNetwork::new(vec![
            Box::new(Reshape::new(d_model, seq_len)),
            Box::new(TransformerBlock::new(d_model, h, d_ff, 0.1, false)),
            Box::new(Flatten),
        ]);

        let dummy_data = vec![
            (vec![0.1; d_model * seq_len], vec![0.2; d_model * seq_len]),
        ];

        let mut optimizer = Adam::new(0.01);
        println!("\n--- Transformer Deneme ---");
        let history = model.train(&dummy_data, 20, 1, &mut optimizer);
        
        assert_eq!(history.len(), 20);
        assert!(history[history.len()-1] <= history[0]); // Kayıp azalmalı
    }

    #[test]
    fn test_tiny_shakespeare_gpt() {
        let text = "ROMEO: Shall I believe that unsubstantial death is amorous?";
        let tokenizer = CharTokenizer::new(text);
        let vocab_size = tokenizer.vocab_size;
        let d_model = 16;
        let n_heads = 2;
        let d_ff = 32;
        let n_layers = 2;
        let max_seq_len = 32;
        
        let mut gpt = TinyShakespeareGPT::new(vocab_size, d_model, n_heads, d_ff, n_layers, max_seq_len, tokenizer);
        let mut optimizer = Adam::new(0.01);
        
        println!("\n--- GPT İlk Deneme ---");
        gpt.train_on_text(text, 200, max_seq_len, &mut optimizer);
        
        let generated = gpt.generate("ROMEO:", 50, 0.8);
        println!("\nUretilen Metin: {}", generated);
        assert!(generated.len() > 6);
    }
}
