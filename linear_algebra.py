import numpy as np
from scipy.linalg import det
from sympy import Matrix
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import streamlit as st


def parsing_input(input_str):
    try:
        input_str = input_str.strip().replace('[', '').replace(']', '')
        rows = input_str.split('\n')
        matrix = []
        for row in rows:
            if row:
                matrix.append([float(num) for num in row.split(',')])
        return matrix
    except ValueError:
        raise ValueError("Input contains non-numeric values.")
    except Exception as e:
        raise ValueError(f"Error parsing matrix: {e}")

def format_input(A_input, b_input):
    try:
        A_matrix = parsing_input(A_input)
        b_matrix = parsing_input(b_input)
        
        if len(A_matrix) != len(b_matrix):
            raise ValueError("Matrix A and matrix b must have the same number of rows.")
        if len(b_matrix[0]) != 1:
            raise ValueError("Matrix b must be a column vector.")
        if len(A_matrix[0]) != len(b_matrix):
            raise ValueError("Matrix A dimensions do not match the number of rows in matrix b.")
        
        for matrix in [A_matrix, b_matrix]:
            for row in matrix:
                for value in row:
                    if not isinstance(value, (int, float)):
                        raise ValueError("Elements must be numbers.")
        return A_matrix, b_matrix
    except ValueError as e:
        st.error(f"Error: {e}")
        return None, None

def is_unique(A, b):
    A_matrix = Matrix(A)
    b_matrix = Matrix(b)
    augmented_matrix = A_matrix.row_join(b_matrix)
    rank_A = A_matrix.rank()
    rank_augmented = augmented_matrix.rank()
    column = A_matrix.shape[1]
    row = A_matrix.shape[0]
    if column == row:
        if A_matrix.det() != 0 and rank_A == rank_augmented:
            return True
    elif column > row:
        return False
    elif row > column:
        if rank_A == rank_augmented:
            return True
    return False

def is_infinite(A, b):
    A_matrix = Matrix(A)
    b_matrix = Matrix(b)
    augmented_matrix = A_matrix.row_join(b_matrix)
    rank_A = A_matrix.rank()
    rank_augmented = augmented_matrix.rank()
    column = A_matrix.shape[1]
    if rank_A != rank_augmented:
        return False
    elif rank_A < column:
        if rank_A == rank_augmented:
            return True
    else:
        return False

def infinite_homo(A, b):
    A_matrix = Matrix(A)
    if is_infinite(A, b):
        null_space_aug = A_matrix.nullspace()
        return null_space_aug
    return []

def solve_square(A, b):
    try:
        A_matrix = Matrix(A)
        b_matrix = Matrix(b)
        augmented_matrix = A_matrix.row_join(b_matrix)
        aug = augmented_matrix.rref()[0]
        size = aug.shape[1]
        vector = aug.col(size - 1)
        if A_matrix.det() == 0:
            raise Exception("System is inconsistent")
        return vector
    except Exception as e:
        return str(e)

def plot_solution(A, b, solution):
    A_matrix = np.array(A)
    b_matrix = np.array(b)
    num_rows, num_cols = A_matrix.shape

    if num_cols == 3: 
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(num_rows):
            a, b, c = A_matrix[i]
            d = b_matrix[i]
            xx, yy = np.meshgrid(range(-10, 10), range(-10, 10))
            z = (d - a * xx - b * yy) / c
            ax.plot_surface(xx, yy, z, alpha=0.5, rstride=100, cstride=100, cmap='viridis')
        if isinstance(solution, list):
            for vector in solution:
                v = np.array(vector, dtype=np.float64)
                ax.quiver(0, 0, 0, v[0], v[1], v[2], length=1.0, color='r')
        else:
            v = np.array(solution, dtype=np.float64)
            ax.quiver(0, 0, 0, v[0], v[1], v[2], length=1.0, color='r')
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])
        ax.set_zlim([-10, 10])
        st.pyplot(fig)

    elif num_cols == 2:  
        fig, ax = plt.subplots()
        for i in range(num_rows):
            a, b = A_matrix[i]
            d = b_matrix[i]
            x = np.linspace(-10, 10, 400)
            y = (d - a * x) / b
            ax.plot(x, y, label=f'Line {i + 1}')
        if isinstance(solution, list):
            for vector in solution:
                v = np.array(vector, dtype=np.float64)
                ax.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='r')
        else:
            v = np.array(solution, dtype=np.float64)
            ax.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='r')
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

st.title('Homogeneous and Non-Homogeneous System Solver')

num_rows = st.number_input('Number of Rows', min_value=1, value=2)
num_cols = st.number_input('Number of Columns', min_value=1, value=2)

A_input = st.text_area(f'Matrix A ({num_rows}x{num_cols})', 
                       value='\n'.join(['[' + ', '.join(['0'] * num_cols) + ']'] * num_rows))

b_input = st.text_area(f'Matrix b ({num_rows}x1)', 
                       value='\n'.join(['[0]'] * num_rows))

if st.button('Solve'):
    A, b = format_input(A_input, b_input)
    
    if A and b:
        if is_unique(A, b):
            st.write("The system has a unique solution.")
            solution = solve_square(A, b)
            st.write("Solution:", solution)
            plot_solution(A, b, solution)
        
        elif is_infinite(A, b):
            st.write("The system has infinite solutions.")
            null_space = infinite_homo(A, b)
            st.write("Null space basis vectors:")
            for vector in null_space:
                st.write(vector)
            plot_solution(A, b, null_space)
        
        else:
            st.write("The system is inconsistent.")
