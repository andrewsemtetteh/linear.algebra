# import libraries for the application
import streamlit as st  # streamlit library for building interactive web applications
import numpy as np  # numpy library for numerical operations and array manipulation
import matplotlib.pyplot as plt  # matplotlib library for creating static, animated, and interactive visualizations
from mpl_toolkits.mplot3d import Axes3D  # mpl_toolkits.mplot3d module for 3D plotting
from scipy.linalg import null_space  # scipy.linalg module for linear algebra operations, specifically for computing null space
import pandas as pd  # pandas library for data manipulation and analysis

# core functions and logic

def parse_matrix_input(matrix_df):
    """convert the dataframe to a numpy array."""
    return matrix_df.values.astype(float)  # convert dataframe values to float and return as numpy array

def parse_vector_input(vector_df):
    """convert the dataframe to a numpy array (flattened)."""
    return vector_df.values.flatten()  # flatten the dataframe values and return as a numpy array

def solve_systems(A, b):
    """solve both homogeneous (Ax = 0) and non-homogeneous (Ax = b) systems."""
    homogeneous_sol = null_space(A)  # compute the null space of matrix A

    try:
        # check if A is square and nonsingular for unique solutions
        if np.linalg.matrix_rank(A) == A.shape[1]:
            # solve non-homogeneous system using least-squares method
            particular_sol = np.linalg.lstsq(A, b, rcond=None)[0]
            # check if the particular solution satisfies the system
            if np.allclose(np.dot(A, particular_sol), b):
                non_homogeneous_sol = particular_sol.reshape(-1, 1) + homogeneous_sol
            else:
                non_homogeneous_sol = None
        else:
            non_homogeneous_sol = None

    except np.linalg.LinAlgError as e:
        non_homogeneous_sol = None
        st.error(f"An error occurred while solving the system: {e}")
    
    return homogeneous_sol, non_homogeneous_sol  # return both solutions

def plot_3d_line(ax, direction, origin, color='b', label='Line'):
    """plot a line in 3D space."""
    t = np.linspace(-10, 10, 100)  # create an array of values for parameter t
    x = origin[0] + direction[0] * t  # compute x coordinates of the line
    y = origin[1] + direction[1] * t  # compute y coordinates of the line
    z = origin[2] + direction[2] * t  # compute z coordinates of the line
    ax.plot(x, y, z, color=color, label=label)  # plot the line in 3D space

def plot_3d_plane(ax, normal, point, color='g', alpha=0.5, label='Plane'):
    """plot a plane in 3D space."""
    d = -point.dot(normal)  # compute the plane's constant term
    xx, yy = np.meshgrid(np.linspace(-10, 10, 10), np.linspace(-10, 10, 10))  # create a grid for the plane
    zz = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]  # compute z coordinates of the plane
    ax.plot_surface(xx, yy, zz, color=color, alpha=alpha, label=label)  # plot the plane in 3D space

def plot_3d_solution(solution, title):
    """plot the solution in 3D space. handles different cases: lines, planes, or no solution."""
    fig = plt.figure(figsize=(8, 6))  # create a new figure
    ax = fig.add_subplot(111, projection='3d')  # add a 3D subplot
    
    if solution is not None and solution.size > 0:
        if solution.shape[1] == 1:  # if the solution has one dimension (line)
            origin = np.zeros(3)  # define the origin of the line
            direction = solution[:, 0]  # extract the direction vector
            plot_3d_line(ax, direction, origin, color='b', label='Solution Line')  # plot the solution line
        elif solution.shape[1] == 2:  # if the solution has two dimensions (plane)
            origin = np.zeros(3)  # define the origin of the plane
            direction1 = solution[:, 0]  # extract the first direction vector
            direction2 = solution[:, 1]  # extract the second direction vector
            normal = np.cross(direction1, direction2)  # compute the normal vector of the plane
            plot_3d_plane(ax, normal, origin, color='g', alpha=0.5, label='Solution Plane')  # plot the solution plane
        elif solution.shape[1] == 3:  # if the solution has three dimensions (point)
            ax.scatter(solution[0, 0], solution[1, 0], solution[2, 0], c='r', marker='o', label='Solution Point')  # plot the solution point
        else:
            ax.text(0, 0, 0, "Solution not visualizable", fontsize=12)  # display a message if the solution cannot be visualized
    else:
        ax.text(0, 0, 0, "No solution", fontsize=12)  # display a message if there is no solution
    
    ax.set_title(title)  # set the title of the plot
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')  # set the labels for the axes
    ax.legend()  # add a legend to the plot
    return fig  # return the figure

def render_editable_table(df):
    """render an editable dataframe as html without row and column headers, with values as integers."""
    df = df.astype(int)  # convert dataframe values to integer
    st.write(
        df.to_html(index=False, header=False, classes='editable-table'),  # convert dataframe to HTML without headers
        unsafe_allow_html=True
    )

# streamlit interface code
def main():
    st.title("Homogeneous and Non-Homogeneous Systems Solver")  # set the title of the streamlit app
    
    st.header("Choose the Size of the Matrix")  # header for matrix size selection
    rows = st.number_input("Number of rows", min_value=1, value=3, step=1)  # input for number of rows
    cols = st.number_input("Number of columns", min_value=1, value=3, step=1)  # input for number of columns
    
    st.header("Enter Matrix A")  # header for matrix A input
    matrix_data = np.zeros((rows, cols))  # initialize matrix with zeros
    matrix_df = pd.DataFrame(matrix_data)  # convert matrix to dataframe
    
    st.write("Matrix A (Enter the Matrix Values)")  # prompt for matrix A values
    matrix_df = st.data_editor(matrix_df, use_container_width=True)  # create an editable dataframe for matrix A
    
    st.write("Matrix A (Output):")  # display matrix A output
    render_editable_table(matrix_df)  # render matrix A as HTML
    
    st.header("Enter Vector b")  # header for vector b input
    b_data = np.zeros(rows)  # initialize vector b with zeros
    b_df = pd.DataFrame(b_data, columns=[''])  # convert vector b to dataframe
    
    st.write("Vector b (Enter the values of Vector b)")  # prompt for vector b values
    b_df = st.data_editor(b_df, use_container_width=True)  # create an editable dataframe for vector b
    
    st.write("Vector b (Output):")  # display vector b output
    render_editable_table(b_df)  # render vector b as HTML
    
    if st.button("Solve"):  # button to trigger solution
        try:
            A = parse_matrix_input(matrix_df)  # parse matrix A from dataframe
            b = parse_vector_input(b_df)  # parse vector b from dataframe
            
            # check if the number of rows in A matches the length of vector b
            if A.shape[0] != b.shape[0]:
                st.error("The number of rows in matrix A must match the number of entries in vector b.")
                return
            
            homogeneous_sol, non_homogeneous_sol = solve_systems(A, b)  # solve the systems
            
            st.subheader("Homogeneous solution (Ax = 0):")  # header for homogeneous solution
            st.write(homogeneous_sol)  # display homogeneous solution
            
            st.subheader("Non-homogeneous solution (Ax = b):")  # header for non-homogeneous solution
            st.write(non_homogeneous_sol if non_homogeneous_sol is not None else "No solution")  # display non-homogeneous solution
            
            st.subheader("Visualizations")  # header for visualizations
            
            st.write("Homogeneous Solution")  # prompt for homogeneous solution visualization
            fig_homogeneous = plot_3d_solution(homogeneous_sol, "Homogeneous Solution")  # plot homogeneous solution
            st.pyplot(fig_homogeneous)  # display plot
            
            st.write("Non-homogeneous Solution")  # prompt for non-homogeneous solution visualization
            fig_non_homogeneous = plot_3d_solution(non_homogeneous_sol, "Non-Homogeneous Solution")  # plot non-homogeneous solution
            st.pyplot(fig_non_homogeneous)  # display plot
            
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")  # handle any unexpected errors

# run the main function to start the streamlit app
if __name__ == "__main__":
    main()
