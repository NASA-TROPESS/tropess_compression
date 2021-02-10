"""
Compress arbitrary MUSES matrices for multiple soundings. 

Copyright 2020, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology.
"""

import numpy as np
import struct 
from timeit import default_timer as timer
from tqdm import tqdm
from bitarray import bitarray



class Multiple_Sounding_Transformation:
    
    def __init__(self, data_array=None, fill_value=-999.0, transformation_mode=1):
        """
        Takes in multiple data matrices as a 3D numpy array where first dimension is 
        the sounding index and second two dimensions are the sounding's number of 
        pressure levels. Bad values are -999.0.
        e.g. (2565, 67, 67)
        """
        #Ensure the data_array is 3-dimensional. 
        #Should put in some checks to make sure it is composed of square matrices. 
        if len(data_array.shape) == 2: 
            self.data_array = np.array([data_array])
        else: 
            self.data_array = data_array 
        
        self.fill_value = fill_value
        self.transformation_mode = transformation_mode 
        self.num_soundings = data_array.shape[0]
        self.orig_dim = data_array.shape[-1]
    
    
    def compute_support_indices(self):
        """
        Compute the indices of the support rows/columns. 
        Saves supp_inds_mat, a (num_soundings x matrix_dim) matrix with 1's corresponding to the support. 
        Also saves num_supp_inds, a vector of length num_soundings with the number of support indices for each sounding. 
        """
        num_soundings = int(self.num_soundings)
        orig_dim = int(self.orig_dim)
        supp_inds_mat = np.zeros((num_soundings, orig_dim,))
        for i_sounding in range(num_soundings):
            #Must take the union of the support indices of each row: 
            for i_row in range(orig_dim):
                supp_inds_mat[i_sounding, np.where(self.data_array[i_sounding, i_row, :] != self.fill_value)[0]] = 1
        num_supp_inds = np.sum(supp_inds_mat, axis=1)
        
        self.supp_inds_mat = supp_inds_mat
        self.num_supp_inds = num_supp_inds
    
    
    def fill_with_zero(self):
        """
        Replace data array fill values with zero. 
        """
        data_array_filled = np.zeros(self.data_array.shape)
        data_array_filled[:, :, :] = self.data_array[:, :, :]
        data_array_filled[np.where(self.data_array == self.fill_value)] = 0
        
        self.data_array_filled = data_array_filled
        
    def find_complete_row_set(self, M):
        """
        Returns a 0,1 vector of length M.shape[1] with 1's corresponding to the rows which form a basis for the row space of M. 
        """
        row_ind_vec = np.zeros((M.shape[0],))
        M_reduced = np.zeros(M.shape)
        M_reduced[:] = M[:]
        row_unit_vec = np.zeros((M.shape[1],))
        for i in range(M.shape[1]):
            row_norms = np.linalg.norm(M_reduced, axis=1)
            row_ind = np.argmax(row_norms)
            row_ind_vec[row_ind] = 1
            row_unit_vec[:] = M_reduced[row_ind, :]/row_norms[row_ind]
            for i_row in range(M_reduced.shape[0]):
                M_reduced[i_row, :] = M_reduced[i_row, :] - np.dot(M_reduced[i_row, :], row_unit_vec) * row_unit_vec

        return(row_ind_vec) 
    
    def construct_transformation_matrices(self):
        """
        Construct matrices T_left and T_right by which to pre- and post-multiply data_array matrices. 
        """
        transformation_mode = self.transformation_mode
        
        if transformation_mode == 1: 
            if not hasattr(self, 'data_array_filled'):
                self.fill_with_zero()
            num_soundings = int(self.num_soundings)
            data_array_filled = self.data_array_filled
            dat_concat_tall = np.concatenate([data_array_filled[i, :, :] for i in range(num_soundings)], axis=0)
            dat_concat_long = np.concatenate([data_array_filled[i, :, :] for i in range(num_soundings)], axis=1)

            U_tall, s_tall, Vh_tall = np.linalg.svd(dat_concat_tall, full_matrices=False)
            U_long, s_long, Vh_long = np.linalg.svd(dat_concat_long, full_matrices=False)

            T_left = U_long.transpose()
            T_right = Vh_tall.transpose()

            #Form a 0,1 matrix whose n^th row lists the row-indices of T_left which form an invertible submatrix
            #when restricted to columns n:-1. 
            T_left_row_sets = np.zeros((T_left.shape[1], T_left.shape[0],))
            for i in range(T_left.shape[0]):
                T_left_row_sets[i, :] = self.find_complete_row_set(T_left[:, i:])

            #Form a 0,1 matrix whose n^th row lists the column-indices of T_right which form an invertible submatrix
            #when restricted to rows n:-1. 
            T_right_col_sets = np.zeros(T_right.shape)
            for i in range(T_right.shape[0]):
                T_right_col_sets[i, :] = self.find_complete_row_set(T_right[i:, :].transpose()) 
        
        elif transformation_mode == 2:
            if not hasattr(self, 'data_array_filled'):
                self.fill_with_zero()
            num_soundings = int(self.num_soundings)
            data_array_filled = self.data_array_filled
            
            #Use this as baseline: 
            dat_concat_long = np.concatenate([data_array_filled[i, :, :] for i in range(num_soundings)], axis=1)
            U_long, s_long, Vh_long = np.linalg.svd(dat_concat_long, full_matrices=False)
            T_left = U_long.transpose()
            T_right = np.zeros(T_left.transpose().shape)
            T_right[:, :] = T_left.transpose()[:, :]
            
            #dat_concat_tall = np.concatenate([data_array_filled[i, :, :] for i in range(num_soundings)], axis=0)
            #U_tall, s_tall, Vh_tall = np.linalg.svd(dat_concat_tall, full_matrices=False)
            #T_right = Vh_tall.transpose()
            #T_left = np.zeros(T_right.transpose().shape)
            #T_left[:, :] = T_right.transpose()[:, :] 
            
            
            #Form a 0,1 matrix whose n^th row lists the row-indices of T_left which form an invertible submatrix
            #when restricted to columns n:-1. 
            T_left_row_sets = np.zeros((T_left.shape[1], T_left.shape[0],))
            for i in range(T_left.shape[0]):
                T_left_row_sets[i, :] = self.find_complete_row_set(T_left[:, i:])

            #Form a 0,1 matrix whose n^th row lists the column-indices of T_right which form an invertible submatrix
            #when restricted to rows n:-1. 
            T_right_col_sets = np.zeros(T_right.shape)
            T_right_col_sets[:, :] = T_left_row_sets[:, :]
        
        elif transformation_mode == 3 or transformation_mode == 4:
            if not hasattr(self, 'data_array_filled'):
                self.fill_with_zero()
            num_soundings = int(self.num_soundings)
            data_array_filled = self.data_array_filled
            
            orig_dim = int(self.orig_dim)
            
            T_left = np.eye(orig_dim)
            T_right = np.eye(orig_dim) 
            
            T_left_row_sets = np.ones((orig_dim, orig_dim,))
            T_right_col_sets = np.ones((orig_dim, orig_dim,))
            
        else:
            print("Unknown transformation mode. ")
            return 
        
        self.T_left = T_left
        self.T_right = T_right 
        self.T_left_row_sets = T_left_row_sets
        self.T_right_col_sets = T_right_col_sets 
    
    
    def apply_transformation(self):
        """
        Transform each individual matrix to prepare for compression. 
        """
        self.construct_transformation_matrices() #Produces data_array_filled, T_left, and T_right 
        data_array_filled = self.data_array_filled
        T_left = self.T_left
        T_right = self.T_right
        T_left_row_sets = self.T_left_row_sets
        T_right_col_sets = self.T_right_col_sets 
        
        if not hasattr(self, 'num_supp_inds'):
            self.compute_support_indices() #Produces num_supp_inds
        
        orig_dim = int(self.orig_dim)
        fill_value = self.fill_value 
        num_soundings = int(self.num_soundings)
        supp_inds_mat = self.supp_inds_mat
        num_supp_inds = self.num_supp_inds
        data_array_transformed = np.zeros(self.data_array.shape) + fill_value
        
        for i_sounding in range(num_soundings):
            ns = int(num_supp_inds[i_sounding])
            if ns > 0:
                
                #MATRIX IS IN THE FORM [L_11, L_12; L_21, L_22] [0, 0; 0, M] [R_11, R_12; R_21, R_22] 
                #= [L_12; L_22] M [R_21, R_22], SO WE ONLY STORE ROWS AND COLUMNS WHICH FORM INVERTIBLE 
                #SUBSETS OF THE ROWS OF [L_12; L_22] AND THE COLUMNS OF [R_21, R_22]. 
                
                if np.all(supp_inds_mat[i_sounding, -ns:] == 1):
                    ind_Tleftrows = np.where(T_left_row_sets[-ns, :] == 1)[0]
                    ind_Trightcols = np.where(T_right_col_sets[-ns, :] == 1)[0]
                    ind_datasuppinds = np.where(np.matmul(T_left_row_sets[-ns, :].reshape((orig_dim, 1)), T_right_col_sets[-ns, :].reshape((1, orig_dim))) == 1)
                    
                    data_array_transformed[i_sounding, ind_datasuppinds[0], ind_datasuppinds[1]] = np.matmul( T_left[ind_Tleftrows, -ns:] , np.matmul( data_array_filled[i_sounding, -ns:, -ns:], T_right[-ns:, ind_Trightcols])).flatten()
                else:
                    data_array_transformed[i_sounding, :, :] = np.matmul( T_left , np.matmul( data_array_filled[i_sounding, :, :], T_right))[:, :]
                    
        return data_array_transformed, T_left, T_right, T_left_row_sets, T_right_col_sets, supp_inds_mat  
    
    
    
    
class Multiple_Sounding_Compression:
    
    def __init__(self, data_array=None, fill_value=-999.0):
        """
        Takes in multiple data matrices as a 3D numpy array where first dimension is 
        the sounding index and second two dimensions are the sounding's number of 
        pressure levels. Bad values are -999.0.
        e.g. (2565, 67, 67)
        """
        #Ensure the data_array is 3-dimensional. 
        #Should put in some checks to make sure it is composed of square matrices. 
        if len(data_array.shape) == 2: 
            self.data_array = np.array([data_array])
        else: 
            self.data_array = data_array 
        
        self.fill_value = fill_value
        self.num_soundings = data_array.shape[0]
        self.orig_dim = data_array.shape[-1]
    
    
    def compute_support_indices_2D(self, arr_2D):
        """
        Compute the indices of the support rows/columns for a single (square) matrix. 
        Assumes support columns are the same as support rows. 
        """
        orig_dim = arr_2D.shape[0]
        supp_inds_vec = np.zeros((orig_dim,))
        for i_row in range(orig_dim):
            supp_inds_vec[np.where(arr_2D[i_row, :] != self.fill_value)[0]] = 1
        
        return supp_inds_vec 
        
    

    def compute_compression_parameters_1D(self, arr, abs_error=0.0001):
        """
        Determine the optimal quotient divisor for maximum Rice-Golomb compression 
        subject to the absolute error tolerance. Then compute the corresponding 
        and Rice-Golomb encoding parameters. Returns a single q_divisor, num_r. 
        """
        
        #For each array entry, we will store a single bit for the sign, followed by bits 
        #encoding the absolute value: the integer part of the quotient by q_divisor (a sequence 
        #of 1's followed by a 0) and a fixed number of bits for the remainder. 
        
        #We make the following estimate, modeling the distribution as Gaussian and 
        #making a few approximations. 
        q_divisor = max((2*np.log(2) / np.sqrt(2*np.pi)) * np.std(arr.flatten()), abs_error) 
        
        #Number of possible quantized remainders: 
        num_r = np.ceil(q_divisor/abs_error)
        
        return q_divisor, num_r
        
        
    def set_compression_parameters_3D(self, arr_3D, abs_error=0.0001):
        """
        Compute the compression parameters for each matrix coordinate. 
        We will compress each coordinate based on the distribution of the corresponding 
        coordinate values for all soundings. Saves 2-D matrices for q_divisor values, 
        num_r values,  mean values, and std values. 
        """
        
        num_soundings = int(self.num_soundings)
        orig_dim = int(self.orig_dim)
        q_divisor_mat = np.zeros((orig_dim, orig_dim,))
        num_r_mat = np.zeros((orig_dim, orig_dim,))
        filtered_mean_mat = np.zeros((orig_dim, orig_dim,))
        
        #Create a dictionary of all possible remainder bit strings. 
        #This significantly speeds up compression in practice. 
        r_bitarray_dict = {}
        
        for i in range(orig_dim):
            for j in range(orig_dim):
                arr = arr_3D[:, i, j]
                ind_arr = np.where(arr != self.fill_value)
                if len(ind_arr[0]) > 0:
                    arr_filtered = arr[ind_arr]
                    arr_mean = np.mean(arr_filtered) 
                    arr_zeromean = arr_filtered - arr_mean
                    q_divisor, num_r = self.compute_compression_parameters_1D(arr_zeromean, abs_error=abs_error)
                    num_bits_r = int(np.ceil(np.log2(num_r)))
                    q_divisor_mat[i, j] = q_divisor
                    num_r_mat[i, j] = num_r
                    filtered_mean_mat[i, j] = arr_mean
                    r_bitarray_dict[i, j] = {} 
                    for r_val in range(int(num_r)):
                        r_bitarray = bitarray(endian='little')
                        r_bitarray.frombytes(struct.pack('Q', r_val))
                        r_bitarray_dict[i, j][r_val] = r_bitarray[num_bits_r-1::-1]
                    
        
        self.q_divisor_mat = q_divisor_mat
        self.num_r_mat = num_r_mat 
        self.filtered_mean_mat = filtered_mean_mat 
        self.r_bitarray_dict = r_bitarray_dict 
        
                    
    def compute_number_compressed_bits_2D(self, arr_2D):
        """
        Determine how many bits we need for an individual matrix. 
        """
        
        ind_arr = np.where(arr_2D != self.fill_value)
        arr_filtered = arr_2D[ind_arr]
        q_divisor_filtered = self.q_divisor_mat[ind_arr]
        num_r_filtered = self.num_r_mat[ind_arr]
        filtered_mean_filt = self.filtered_mean_mat[ind_arr]
        
        arr_minus_mean_filt = arr_filtered - filtered_mean_filt
        
        num_arr_entries = arr_filtered.shape[0]
        
        #Start with bits for support indices: 
        num_bits_supp_inds = self.orig_dim
        
        #Include a bit for the sign (+/-1): 
        num_bits_sign_total = num_arr_entries
        
        #The bits for the quotients by q_divisor: 
        num_bits_q_total = np.sum((np.abs(arr_minus_mean_filt)) // q_divisor_filtered) + num_arr_entries
        
        #The bits for the remainders: 
        num_bits_r_total = np.sum(np.ceil(np.log2(num_r_filtered)))
        
        #Compute the total number of bits in compressed form: 
        num_bits_compressed_total = num_bits_supp_inds + num_bits_q_total + num_bits_r_total + num_bits_sign_total
        
        return num_bits_compressed_total
        
    
    def compress_2D(self, arr_2D, abs_error=0.0001, supp_inds_vec=[]):
        """
        Perform a variation of Rice-Golomb compression on a single matrix. 
        """
        
        orig_dim = self.orig_dim
        
        if len(supp_inds_vec)==0:
            supp_inds_vec = self.compute_support_indices_2D(arr_2D)
        
        num_bits_compressed = int(self.compute_number_compressed_bits_2D(arr_2D))
        
        ind_arr = np.where(arr_2D != self.fill_value)
        arr_filtered = arr_2D[ind_arr]
        q_divisor_filtered = self.q_divisor_mat[ind_arr]
        num_r_filtered = self.num_r_mat[ind_arr]
        filtered_mean_filt = self.filtered_mean_mat[ind_arr]
        
        arr_minus_mean_filt = arr_filtered - filtered_mean_filt
        
        num_arr_entries = arr_minus_mean_filt.shape[0]
        
        #Create 0,1 array corresponding to the signs. (1 = +, 0 = -) 
        arr_signs = np.sign(arr_minus_mean_filt)
        #arr_signs[arr_signs >= 0] = 1
        arr_signs[arr_signs == 0] = 1  #No need to rewrite correct entries. 
        arr_signs[arr_signs < 0] = 0 
        
        
        #Compute an array of the coarse estimates of the element absolute values, i.e., which multiple of q_divisor. 
        arr_q = ( np.abs(arr_minus_mean_filt) ) // q_divisor_filtered 
        
        #Compute the quantized remainders. 
        arr_r = (np.abs(arr_minus_mean_filt) - (q_divisor_filtered * arr_q)) // abs_error
        
        #Number of bits needed to store remainder. 
        num_bits_r_filtered = np.ceil(np.log2(num_r_filtered))
        
        #Initialize bit array for the compressed version of the 2D array: 
        arr_compressed_01 = bitarray(num_bits_compressed, endian='big') 
        arr_compressed_01.setall(1)
        
        #The format will be as follows: We will store bits corresponding to the support indices 
        #(the indices of included rows, assumed to be the same for columns). 
        #Then, for each filtered array entry, we store a bit for the sign, bits for the quotient, 
        #and bits for the remainder. 
        
        #Keep a variable indexing our position in the bitarray. 
        i_bit = 0
        
        #First, store support indices: 
        arr_compressed_01[i_bit:i_bit + orig_dim] = bitarray([i for i in supp_inds_vec])
        i_bit = int(i_bit + orig_dim)
        
        for i_arr in range(int(num_arr_entries)):
            arr_sign_i = arr_signs[i_arr]
            arr_q_i = arr_q[i_arr]
            arr_r_i = arr_r[i_arr] 
            num_bits_r_i = num_bits_r_filtered[i_arr] 
            
            #Insert sign bit
            if int(arr_compressed_01[i_bit]) != int(arr_sign_i): #Don't rewrite if not necessary
                arr_compressed_01[i_bit] = arr_sign_i
            
            #Advance position by one (sign) bit. 
            i_bit = i_bit + 1
            
            #Insert q bits: ones followed by a terminating zero
            #arr_compressed_01[i_bit:i_bit + int(arr_q_i)] = 1 #Don't rewrite if unnecessary. 
            arr_compressed_01[i_bit + int(arr_q_i)] = 0
            
            #Advance position by the appropriate number of quotient bits. 
            i_bit = i_bit + int(arr_q_i) + 1
            
            #Insert r bits. 
            if num_bits_r_i > 0:
                #Get the proper remainder bits from the proper stored dictionary:
                i_dict = ind_arr[0][i_arr]
                j_dict = ind_arr[1][i_arr]
                arr_compressed_01[i_bit:i_bit + int(num_bits_r_i)] = self.r_bitarray_dict[i_dict, j_dict][int(arr_r_i)]
                
            #Advance position by the appropriate number of remainder bits. 
            i_bit = i_bit + int(num_bits_r_i) 
            
        arr_compressed_bytes = arr_compressed_01.tobytes()
        
        return(arr_compressed_bytes)

    
    def remove_bad_soundings(self, compression_mode):
        """
        Function to remove soundings deemed too uncharacteristic to compress. 
        """
        num_soundings = int(self.num_soundings)
        orig_dim = int(self.orig_dim)
        fill_value = self.fill_value
        
        if compression_mode==1:
            vv_temp = np.zeros((orig_dim,))
            for i in range(num_soundings):
                vv_temp[:] = np.diag(self.data_array[i, :, :])
                ind_temp = np.where(vv_temp != fill_value)
                if np.any(np.abs(vv_temp[ind_temp]) > 3.0):
                    self.data_array[i, :, :] = fill_value
        
            
        
        
    
    
    
    def compress_3D(self, max_error=0.00005, compression_mode=1):
        """
        Compress 3D array, with first dimension corresponding to number of soundings, and 
        2nd and 3rd dimensions equal. 
        transform_mode: 0 (Do not transform), 1 (Transform), 2 (Transform, matrices are symmetric)
        """
        
        abs_error = 2*max_error 
        
        if compression_mode == 1 or compression_mode == 3:
            
            #Specify how we flag soundings which we will not compress for a given compression mode. 
            self.remove_bad_soundings(compression_mode=compression_mode)
            
            MST = Multiple_Sounding_Transformation(self.data_array, transformation_mode=compression_mode)
            arr_3D, T_left, T_right, T_left_row_sets, T_right_col_sets, supp_inds_mat = MST.apply_transformation()
            #arr_3D contains the transformed data array of size self.num_soundings x self.orig_dim x self.orig_dim
            #T_left and T_right are square transformation matrices of size self.orig_dim x self.orig_dim
            #T_left_row_sets contains the rows of T_left used to invert the stored data (orig_dim x orig_dim)
            #T_right_col_sets contains the cols of T_right used to invert the stored data (orig_dim x orig_dim) 
            #supp_inds_mat is num_soundings x orig_dim, 0,1 matrix with 1's in support indices for each sounding. 
            
            self.set_compression_parameters_3D(arr_3D=arr_3D, abs_error=abs_error)

            orig_dim = int(self.orig_dim)
            num_soundings = int(self.num_soundings) 
            q_divisor_mat = self.q_divisor_mat
            num_r_mat = self.num_r_mat 
            filtered_mean_mat = self.filtered_mean_mat 

            #Store 1 byte for the compression mode, four bytes for the original dimension, 8 bytes for the num_soundings, 
            #8 bytes for abs_error, 8*orig_dim*orig_dim bytes for each of T_left and T_right, 
            #8*orig_dim*orig_dim bytes for each of the filtered_mean_mat, q_divisor_mat, and num_r_mat, 
            #Followed by 8*num_soundings bytes for the starting byte locations of each sounding, 
            #Followed by the bytes for each sounding. 
            
            compression_mode_byte = struct.pack('B', int(compression_mode))
            orig_dim_bytes = struct.pack('I', orig_dim)
            num_soundings_bytes = struct.pack('Q', num_soundings)
            abs_error_bytes = struct.pack('d', abs_error)

            compressed_byte_list = []
            compressed_byte_list.append(compression_mode_byte)
            compressed_byte_list.append(orig_dim_bytes)
            compressed_byte_list.append(num_soundings_bytes)
            compressed_byte_list.append(abs_error_bytes)

            T_left_byte_list = []
            T_right_byte_list = []
            filtered_mean_mat_byte_list = []
            q_divisor_mat_byte_list = []
            num_r_mat_byte_list = []
            
            for i in range(orig_dim):
                for j in range(orig_dim):
                    T_left_bytes = struct.pack('d', T_left[i, j])
                    T_left_byte_list.append(T_left_bytes)

                    T_right_bytes = struct.pack('d', T_right[i, j])
                    T_right_byte_list.append(T_right_bytes)

                    filtered_mean_mat_bytes = struct.pack('d', filtered_mean_mat[i, j])
                    filtered_mean_mat_byte_list.append(filtered_mean_mat_bytes)

                    q_divisor_mat_bytes = struct.pack('d', q_divisor_mat[i, j])
                    q_divisor_mat_byte_list.append(q_divisor_mat_bytes)

                    num_r_mat_bytes = struct.pack('Q', int(num_r_mat[i, j]))
                    num_r_mat_byte_list.append(num_r_mat_bytes)
            
            
            T_left_row_sets_bitarray = bitarray([i == 1 for i in T_left_row_sets.flatten()], endian='big')
            T_left_row_sets_byte_list = [T_left_row_sets_bitarray.tobytes()]
            
            T_right_col_sets_bitarray = bitarray([i == 1 for i in T_right_col_sets.flatten()], endian='big')
            T_right_col_sets_byte_list = [T_right_col_sets_bitarray.tobytes()] 
            
            compressed_byte_list += T_left_byte_list
            compressed_byte_list += T_right_byte_list
            compressed_byte_list += T_left_row_sets_byte_list
            compressed_byte_list += T_right_col_sets_byte_list 
            compressed_byte_list += filtered_mean_mat_byte_list
            compressed_byte_list += q_divisor_mat_byte_list
            compressed_byte_list += num_r_mat_byte_list 
            

            #Compute current number of compressed bytes. 
            num_compressed_bytes_total = 0
            for i in range(len(compressed_byte_list)):
                num_compressed_bytes_total = num_compressed_bytes_total + len(compressed_byte_list[i])

            #Shift ahead by 8 bytes for each sounding to store its index. 
            num_compressed_bytes_total = int(num_compressed_bytes_total + 8*num_soundings)

            #Collect bytes for each individual sounding, and insert bytes for their indices: 
            sounding_byte_list = []
            sounding_byte_ind = num_compressed_bytes_total
            
            for i_sounding in tqdm(range(num_soundings)):
                
                arr_2D = arr_3D[i_sounding, :, :]
                supp_inds_vec = supp_inds_mat[i_sounding, :]
                arr_compressed_bytes = self.compress_2D(arr_2D=arr_2D, abs_error=abs_error, supp_inds_vec=supp_inds_vec)
                sounding_byte_list.append(arr_compressed_bytes)

                sounding_byte_ind_bytes = struct.pack('Q', sounding_byte_ind)
                compressed_byte_list.append(sounding_byte_ind_bytes) 
                
                sounding_byte_ind = int(sounding_byte_ind + len(arr_compressed_bytes))
                
            
            #Append the sounding bytes 
            compressed_byte_list += sounding_byte_list

            #Update number of compressed bytes as the final computed sounding_byte_ind 
            #(index of what would have been the next sounding byte). 
            num_compressed_bytes_total = int(sounding_byte_ind)

            #Now put this all into a byte array: 
            compressed_byte_array = bytearray(int(num_compressed_bytes_total))
            i_byte = 0
            for i in range(len(compressed_byte_list)):
                N_newbytes = len(compressed_byte_list[i])
                compressed_byte_array[i_byte:i_byte + N_newbytes] = compressed_byte_list[i]
                i_byte = i_byte + N_newbytes
            
            return(compressed_byte_array)
        
        
        elif compression_mode == 2 or compression_mode == 4:
            #For symmetric matrices: This largely mirrors the general case (compression_mode = 1), but T_left and T_right are 
            #conjugate, so only one must be stored. Also, we need only store the upper triangular part of the arrays. 
            
            #Specify how we flag soundings which we will not compress for a given compression mode. 
            self.remove_bad_soundings(compression_mode=compression_mode)
            
            MST = Multiple_Sounding_Transformation(self.data_array, transformation_mode=compression_mode)
            arr_3D, T_left, T_right, T_left_row_sets, T_right_col_sets, supp_inds_mat = MST.apply_transformation()
            
            self.set_compression_parameters_3D(arr_3D=arr_3D, abs_error=abs_error)

            orig_dim = int(self.orig_dim)
            fill_value = self.fill_value 
            num_soundings = int(self.num_soundings) 
            q_divisor_mat = self.q_divisor_mat
            num_r_mat = self.num_r_mat 
            filtered_mean_mat = self.filtered_mean_mat 
            
            #Store 1 byte for the compression mode, four bytes for the original dimension, 8 bytes for the num_soundings, 
            #8 bytes for abs_error, 8*orig_dim*orig_dim bytes for only T_left (not T_right), 
            #8*orig_dim*(orig_dim + 1)/2 bytes for each of the filtered_mean_mat, q_divisor_mat, and num_r_mat, 
            #Followed by 8*num_soundings bytes for the starting byte locations of each sounding, 
            #Followed by the bytes for each sounding. 
            
            compression_mode_byte = struct.pack('B', int(compression_mode))
            orig_dim_bytes = struct.pack('I', orig_dim)
            num_soundings_bytes = struct.pack('Q', num_soundings)
            abs_error_bytes = struct.pack('d', abs_error)

            compressed_byte_list = []
            compressed_byte_list.append(compression_mode_byte)
            compressed_byte_list.append(orig_dim_bytes)
            compressed_byte_list.append(num_soundings_bytes)
            compressed_byte_list.append(abs_error_bytes)
            
            T_left_byte_list = []
            filtered_mean_mat_byte_list = []
            q_divisor_mat_byte_list = []
            num_r_mat_byte_list = []
            
            for i in range(orig_dim):
                for j in range(orig_dim):
                    T_left_bytes = struct.pack('d', T_left[i, j])
                    T_left_byte_list.append(T_left_bytes)
            
            #The following only need the upper triangular parts stored. 
            for i in range(orig_dim):
                for j in range(i, orig_dim):
                    filtered_mean_mat_bytes = struct.pack('d', filtered_mean_mat[i, j])
                    filtered_mean_mat_byte_list.append(filtered_mean_mat_bytes)

                    q_divisor_mat_bytes = struct.pack('d', q_divisor_mat[i, j])
                    q_divisor_mat_byte_list.append(q_divisor_mat_bytes)

                    num_r_mat_bytes = struct.pack('Q', int(num_r_mat[i, j]))
                    num_r_mat_byte_list.append(num_r_mat_bytes)
            
            T_left_row_sets_bitarray = bitarray([i == 1 for i in T_left_row_sets.flatten()], endian='big')
            T_left_row_sets_byte_list = [T_left_row_sets_bitarray.tobytes()]
            
            compressed_byte_list += T_left_byte_list
            compressed_byte_list += T_left_row_sets_byte_list
            compressed_byte_list += filtered_mean_mat_byte_list
            compressed_byte_list += q_divisor_mat_byte_list
            compressed_byte_list += num_r_mat_byte_list 
            
            #Compute current number of compressed bytes. 
            num_compressed_bytes_total = 0
            for i in range(len(compressed_byte_list)):
                num_compressed_bytes_total = num_compressed_bytes_total + len(compressed_byte_list[i])

            #Shift ahead by 8 bytes for each sounding to store its index. 
            num_compressed_bytes_total = int(num_compressed_bytes_total + 8*num_soundings)

            #Collect bytes for each individual sounding, and insert bytes for their indices: 
            sounding_byte_list = []
            sounding_byte_ind = num_compressed_bytes_total
            
            for i_sounding in tqdm(range(num_soundings)):
                
                arr_2D = arr_3D[i_sounding, :, :]
                
                #Replace lower-triangular half with fill_value, since we don't need to store it. 
                for j in range(1, orig_dim):
                    for i in range(j, orig_dim):
                        arr_2D[i, j] = fill_value
                
                supp_inds_vec = supp_inds_mat[i_sounding, :]
                arr_compressed_bytes = self.compress_2D(arr_2D=arr_2D, abs_error=abs_error, supp_inds_vec=supp_inds_vec)
                sounding_byte_list.append(arr_compressed_bytes)

                sounding_byte_ind_bytes = struct.pack('Q', sounding_byte_ind)
                compressed_byte_list.append(sounding_byte_ind_bytes) 
                
                sounding_byte_ind = int(sounding_byte_ind + len(arr_compressed_bytes))
                
            
            #Append the sounding bytes 
            compressed_byte_list += sounding_byte_list

            #Update number of compressed bytes as the final computed sounding_byte_ind 
            #(index of what would have been the next sounding byte). 
            num_compressed_bytes_total = int(sounding_byte_ind)

            #Now put this all into a byte array: 
            compressed_byte_array = bytearray(int(num_compressed_bytes_total))
            i_byte = 0
            for i in range(len(compressed_byte_list)):
                N_newbytes = len(compressed_byte_list[i])
                compressed_byte_array[i_byte:i_byte + N_newbytes] = compressed_byte_list[i]
                i_byte = i_byte + N_newbytes
            
            return(compressed_byte_array)
                
        
        else:
            print("Unknown compression mode.")
            return 
        
        
class Multiple_Sounding_Decompression:

    def __init__(self, compressed_data_bytes=None):
        """
        Take in the compressed_data_bytes returned by compress_3D function 
        from the Multiple_Sounding_Compression class. 
        """

        self.compressed_data_bytes = compressed_data_bytes 

    
    def invert_transformation(self, arr_2D):
        """
        Apply the inverse transformation to the 2D array. 
        """
        if arr_2D.shape[0] > 0:
            ind_T_inv = int(self.orig_dim - arr_2D.shape[0])
            T_left_inv = self.T_left_inv_list[ind_T_inv]
            T_right_inv = self.T_right_inv_list[ind_T_inv] 

            arr_2D_invtrans = np.matmul(T_left_inv, np.matmul(arr_2D, T_right_inv)) 
        else: 
            arr_2D_invtrans = np.array([])
        return arr_2D_invtrans 
        
    
    def decompress_2D(self, sounding_bytes, fill_value=-999.0):
        """
        Decompress the bytes for a single sounding. 
        """
        
        if self.compression_mode == 1 or compression_mode == 3:
            
            orig_dim = int(self.orig_dim)
            abs_error = self.abs_error

            sounding_bits = bitarray(endian='big')
            sounding_bits.frombytes(bytes(sounding_bytes))
            
            i_bit = 0

            #Get support index bits: 
            supp_inds_vec = False + np.zeros((orig_dim,))
            for i in range(orig_dim):
                if sounding_bits[i_bit]:
                    supp_inds_vec[i] = True 
                i_bit = i_bit + 1
            
            num_supp_inds = int(np.sum(supp_inds_vec))
            
            if num_supp_inds == 0:
                num_arr_entries = 0
            elif np.all(supp_inds_vec[-num_supp_inds:] == 1):
                num_arr_entries = int(num_supp_inds**2)
                ind_rowcol_sets = int(orig_dim - num_supp_inds)
            else:
                num_arr_entries = int(orig_dim**2) 
                ind_rowcol_sets = 0 
            
            if num_arr_entries > 0:
                Tleftrows_vec = self.T_left_row_sets[ind_rowcol_sets, :]
                Trightcols_vec = self.T_right_col_sets[ind_rowcol_sets, :] 
                arr_2D_supp_mat = np.matmul(Tleftrows_vec.reshape((orig_dim, 1)), Trightcols_vec.reshape((1, orig_dim)))
                ind_arr_2D = np.where(arr_2D_supp_mat == 1)
                
                arr_entries = np.zeros((num_arr_entries,))
                q_divisors_filtered = self.q_divisor_mat[ind_arr_2D]
                num_r_filtered = self.num_r_mat[ind_arr_2D] 
                filtered_mean_filtered = self.filtered_mean_mat[ind_arr_2D]
                
                entry_signs_filtered = np.zeros((num_arr_entries,))
                N_rbits_filtered = np.ceil(np.log2(num_r_filtered))
                N_qbits_filtered = np.zeros((num_arr_entries,)) 
                entry_r_part_initial_filtered = np.zeros((num_arr_entries,)) 
                
                r_bit_array = bitarray(64, endian='little')
            
            
            for i_arr in range(num_arr_entries):
                
                #Get sign of array entry: 
                if sounding_bits[i_bit] == 0:
                    entry_signs_filtered[i_arr] = -1.0 
                else:
                    entry_signs_filtered[i_arr] = 1.0 
                i_bit = i_bit + 1
                
                #Compute the multiple of q_divisor: 
                while sounding_bits[i_bit] == 1:
                    N_qbits_filtered[i_arr] = N_qbits_filtered[i_arr] + 1 #FOR TESTING 
                    i_bit = i_bit + 1
                i_bit = i_bit + 1
                
                
                #Get the r bits 
                N_rbits = int(N_rbits_filtered[i_arr]) 

                #Now the remainder: 
                r_bit_array.setall(0)
                if N_rbits > 0:
                    r_bit_array[:N_rbits] = sounding_bits[i_bit + N_rbits - 1 : i_bit - 1 : -1]
                    i_bit = i_bit + N_rbits
                entry_r_part_initial_filtered[i_arr] = struct.unpack('Q', r_bit_array.tobytes())[0]
                
                
            
            #Create an initial 2D array populated with fill value. 
            arr_2D = fill_value + np.zeros((orig_dim, orig_dim,))
            
            if num_arr_entries > 0:
                
                entry_q_part_filtered = N_qbits_filtered * q_divisors_filtered
                entry_r_part_filtered = abs_error*(entry_r_part_initial_filtered + 0.5)
                entry_value_minus_mean_filtered = entry_signs_filtered * (entry_q_part_filtered + entry_r_part_filtered)
                arr_entries[:] = (entry_value_minus_mean_filtered + filtered_mean_filtered)[:] 
                
                #Fill in nonzero entries of a 2D array. 
                arr_2D_0_dim = int(np.sqrt(num_arr_entries))
                arr_2D_0 = np.zeros((arr_2D_0_dim, arr_2D_0_dim,)) 
                arr_2D_0[:] = arr_entries.reshape((arr_2D_0_dim, arr_2D_0_dim,))[:]

                #Apply inverse transformation: 
                arr_2D_invtrans = self.invert_transformation(arr_2D_0)

                #Reinsert fill value: 
                supp_inds_invtrans = np.zeros((orig_dim, orig_dim,))
                for i in range(orig_dim):
                    if supp_inds_vec[i] == 1:
                        supp_inds_invtrans[i, np.where(supp_inds_vec == 1)[0]] = 1
                ind_arr_2D_invtrans = np.where(supp_inds_invtrans == 1)
                
                arr_2D[ind_arr_2D_invtrans] = arr_2D_invtrans[ind_arr_2D_invtrans] 
            
            return(arr_2D)
        
        
        elif self.compression_mode == 2 or compression_mode == 4:
            
            orig_dim = int(self.orig_dim)
            abs_error = self.abs_error

            sounding_bits = bitarray(endian='big')
            sounding_bits.frombytes(bytes(sounding_bytes))
            
            i_bit = 0

            #Get support index bits: 
            supp_inds_vec = False + np.zeros((orig_dim,))
            for i in range(orig_dim):
                if sounding_bits[i_bit]:
                    supp_inds_vec[i] = True 
                i_bit = i_bit + 1
            
            num_supp_inds = int(np.sum(supp_inds_vec))
            
            if num_supp_inds == 0:
                num_arr_entries = 0
            elif np.all(supp_inds_vec[-num_supp_inds:] == 1):
                num_arr_entries = int((1.0/2.0)*num_supp_inds*(num_supp_inds + 1)) 
                ind_rowcol_sets = int(orig_dim - num_supp_inds)
            else:
                num_arr_entries = int((1.0/2.0)*orig_dim*(orig_dim + 1)) 
                ind_rowcol_sets = 0 
            
            if num_arr_entries > 0:
                Tleftrows_vec = self.T_left_row_sets[ind_rowcol_sets, :]
                Trightcols_vec = self.T_right_col_sets[ind_rowcol_sets, :] 
                #arr_2D_supp_mat = np.matmul(Tleftrows_vec.reshape((orig_dim, 1)), Trightcols_vec.reshape((1, orig_dim)))
                
                arr_2D_supp_mat = np.zeros((orig_dim, orig_dim,))
                for i in range(orig_dim):
                    if Tleftrows_vec[i] == 1:
                        for j in range(i, orig_dim):
                            if Trightcols_vec[j] == 1:
                                arr_2D_supp_mat[i, j] = 1
                
                ind_arr_2D = np.where(arr_2D_supp_mat == 1)
                
                arr_entries = np.zeros((num_arr_entries,))
                q_divisors_filtered = self.q_divisor_mat[ind_arr_2D]
                num_r_filtered = self.num_r_mat[ind_arr_2D] 
                filtered_mean_filtered = self.filtered_mean_mat[ind_arr_2D]
                
                entry_signs_filtered = np.zeros((num_arr_entries,))
                N_rbits_filtered = np.ceil(np.log2(num_r_filtered))
                N_qbits_filtered = np.zeros((num_arr_entries,)) 
                entry_r_part_initial_filtered = np.zeros((num_arr_entries,)) 
                
                r_bit_array = bitarray(64, endian='little')
                
            
            for i_arr in range(num_arr_entries):
                
                #Get sign of array entry: 
                if sounding_bits[i_bit] == 0:
                    entry_signs_filtered[i_arr] = -1.0 
                else:
                    entry_signs_filtered[i_arr] = 1.0 
                i_bit = i_bit + 1
                
                #Compute the multiple of q_divisor: 
                while sounding_bits[i_bit] == 1:
                    N_qbits_filtered[i_arr] = N_qbits_filtered[i_arr] + 1 #FOR TESTING 
                    i_bit = i_bit + 1
                i_bit = i_bit + 1
                
                
                #Get the r bits 
                N_rbits = int(N_rbits_filtered[i_arr]) 

                #Now the remainder: 
                r_bit_array.setall(0)
                if N_rbits > 0:
                    r_bit_array[:N_rbits] = sounding_bits[i_bit + N_rbits - 1 : i_bit - 1 : -1]
                    i_bit = i_bit + N_rbits
                entry_r_part_initial_filtered[i_arr] = struct.unpack('Q', r_bit_array.tobytes())[0]
                
            
            #Create an initial 2D array populated with fill value. 
            arr_2D = fill_value + np.zeros((orig_dim, orig_dim,))
            
            if num_arr_entries > 0:
                
                entry_q_part_filtered = N_qbits_filtered * q_divisors_filtered
                entry_r_part_filtered = abs_error*(entry_r_part_initial_filtered + 0.5)
                entry_value_minus_mean_filtered = entry_signs_filtered * (entry_q_part_filtered + entry_r_part_filtered)
                arr_entries[:] = (entry_value_minus_mean_filtered + filtered_mean_filtered)[:] 
                
                #Fill in nonzero entries of a 2D array. 
                arr_2D_0_dim = int( (np.sqrt(8.0*num_arr_entries + 1.0) - 1.0)/2.0 )
                arr_2D_0 = np.zeros((arr_2D_0_dim, arr_2D_0_dim,)) 
                i_arr = 0
                for i in range(arr_2D_0_dim):
                    for j in range(i, arr_2D_0_dim):
                        arr_2D_0[i, j] = arr_entries[i_arr]
                        if j > i:
                            arr_2D_0[j, i] = arr_entries[i_arr]
                        i_arr = i_arr + 1 
               
                #Apply inverse transformation: 
                arr_2D_invtrans = self.invert_transformation(arr_2D_0)

                #Reinsert fill value: 
                supp_inds_invtrans = np.zeros((orig_dim, orig_dim,))
                for i in range(orig_dim):
                    if supp_inds_vec[i] == 1:
                        supp_inds_invtrans[i, np.where(supp_inds_vec == 1)[0]] = 1
                ind_arr_2D_invtrans = np.where(supp_inds_invtrans == 1)
                
                arr_2D[ind_arr_2D_invtrans] = arr_2D_invtrans[ind_arr_2D_invtrans] 
            
            return(arr_2D)
        

    def invert_transformation_matrices(self, T_left, T_right, T_left_row_sets, T_right_col_sets):
        """
        Construct the matrices by which to left and right multiply the transformed data to invert the transformation: 
        """
        T_left_inv_list = []
        T_right_inv_list = []
        
        for i in range(T_left.shape[1]):
            ind_Tleftrows = np.where(T_left_row_sets[i, :] == 1)[0]
            T_left_inv = np.zeros((T_left.shape[0], int(T_left.shape[1] - i),))
            T_left_inv[i:, :] = np.linalg.inv(T_left[ind_Tleftrows, i:])
            T_left_inv_list.append(T_left_inv)
        
        for j in range(T_right.shape[0]):
            ind_Trightcols = np.where(T_right_col_sets[j, :] == 1)[0]
            T_right_inv = np.zeros((int(T_right.shape[0] - j), T_right.shape[1],))
            T_right_inv[:, j:] = np.linalg.inv(T_right[j:, ind_Trightcols])
            T_right_inv_list.append(T_right_inv) 
        
        return T_left_inv_list, T_right_inv_list
    

    def decompress_3D(self, sounding_indices=None, fill_value=-999.0):
        """
        Decompress the compressed soundings indicated by sounding_indices. 
        """
        
        num_bytes_B = 1
        num_bytes_I = 4
        num_bytes_Q = 8
        num_bytes_d = 8
        
        i_byte = 0
        
        compression_mode = struct.unpack('B', bytes([self.compressed_data_bytes[0]]))[0]
        i_byte = i_byte + num_bytes_B
        
        if compression_mode == 1 or compression_mode == 3:
            orig_dim = struct.unpack('I', self.compressed_data_bytes[i_byte:i_byte + num_bytes_I])[0]
            i_byte = i_byte + num_bytes_I
            
            num_soundings = struct.unpack('Q', self.compressed_data_bytes[i_byte:i_byte + num_bytes_Q])[0]
            i_byte = i_byte + num_bytes_Q
            
            abs_error = struct.unpack('d', self.compressed_data_bytes[i_byte:i_byte + num_bytes_d])[0]
            i_byte = i_byte + num_bytes_d
            
            T_left = np.zeros((orig_dim, orig_dim,))
            T_right = np.zeros((orig_dim, orig_dim,))
            T_left_row_sets = np.zeros((orig_dim, orig_dim,))
            T_right_col_sets = np.zeros((orig_dim, orig_dim,)) 
            filtered_mean_mat = np.zeros((orig_dim, orig_dim,))
            q_divisor_mat = np.zeros((orig_dim, orig_dim,))
            num_r_mat = np.zeros((orig_dim, orig_dim,))
            
            for i in range(orig_dim):
                for j in range(orig_dim):
                    T_left[i, j] = struct.unpack('d', self.compressed_data_bytes[i_byte:i_byte + num_bytes_d])[0]
                    i_byte = i_byte + num_bytes_d
            
            for i in range(orig_dim):
                for j in range(orig_dim):
                    T_right[i, j] = struct.unpack('d', self.compressed_data_bytes[i_byte:i_byte + num_bytes_d])[0]
                    i_byte = i_byte + num_bytes_d        
            
            num_bytes_origdimsqbits = int(np.ceil(orig_dim*orig_dim / 8))
            
            T_left_row_sets_bytes = bytes(self.compressed_data_bytes[i_byte:i_byte + num_bytes_origdimsqbits])
            T_left_row_sets_bits = bitarray(endian='big')
            T_left_row_sets_bits.frombytes(T_left_row_sets_bytes)
            T_left_row_sets = np.zeros((orig_dim, orig_dim,))
            i_bit_temp = 0
            for i in range(orig_dim):
                for j in range(orig_dim):
                    if T_left_row_sets_bits[i_bit_temp]:
                        T_left_row_sets[i, j] = 1
                    i_bit_temp = i_bit_temp + 1 
            i_byte = i_byte + num_bytes_origdimsqbits
            
            T_right_col_sets_bytes = bytes(self.compressed_data_bytes[i_byte:i_byte + num_bytes_origdimsqbits])
            T_right_col_sets_bits = bitarray(endian='big')
            T_right_col_sets_bits.frombytes(T_right_col_sets_bytes)
            T_right_col_sets = np.zeros((orig_dim, orig_dim,))
            i_bit_temp = 0
            for i in range(orig_dim):
                for j in range(orig_dim):
                    if T_right_col_sets_bits[i_bit_temp]:
                        T_right_col_sets[i, j] = 1
                    i_bit_temp = i_bit_temp + 1 
            i_byte = i_byte + num_bytes_origdimsqbits
            
            for i in range(orig_dim):
                for j in range(orig_dim):
                    filtered_mean_mat[i, j] = struct.unpack('d', self.compressed_data_bytes[i_byte:i_byte + num_bytes_d])[0]
                    i_byte = i_byte + num_bytes_d
            
            for i in range(orig_dim):
                for j in range(orig_dim):
                    q_divisor_mat[i, j] = struct.unpack('d', self.compressed_data_bytes[i_byte:i_byte + num_bytes_d])[0]
                    i_byte = i_byte + num_bytes_d
            
            for i in range(orig_dim):
                for j in range(orig_dim):
                    num_r_mat[i, j] = struct.unpack('Q', self.compressed_data_bytes[i_byte:i_byte + num_bytes_Q])[0]
                    i_byte = i_byte + num_bytes_d
            
            sounding_byte_ind_list = []
            for i in range(num_soundings):
                sounding_byte_ind = struct.unpack('Q', self.compressed_data_bytes[i_byte:i_byte + num_bytes_Q])[0]
                sounding_byte_ind_list.append(sounding_byte_ind)
                i_byte = i_byte + num_bytes_Q 
            
            
            #Invert transformation matrices: 
            T_left_inv_list, T_right_inv_list = self.invert_transformation_matrices(T_left, T_right, T_left_row_sets, T_right_col_sets) 
            
            self.compression_mode = compression_mode
            self.orig_dim = orig_dim
            self.num_soundings = num_soundings
            self.abs_error = abs_error
            self.T_left = T_left
            self.T_right = T_right
            self.T_left_row_sets = T_left_row_sets
            self.T_right_col_sets = T_right_col_sets
            self.T_left_inv_list = T_left_inv_list
            self.T_right_inv_list = T_right_inv_list 
            self.filtered_mean_mat = filtered_mean_mat
            self.q_divisor_mat = q_divisor_mat
            self.num_r_mat = num_r_mat
            self.sounding_byte_ind_list = sounding_byte_ind_list 
            
            if sounding_indices==None:
                sounding_indices = [i for i in range(num_soundings)]
            
            arr_3D = np.zeros((len(sounding_indices), orig_dim, orig_dim,))
            
            for i in tqdm(range(len(sounding_indices))): 
                sounding_ind = int(sounding_indices[i])
                sounding_byte_ind = sounding_byte_ind_list[sounding_ind]
                if sounding_ind == num_soundings - 1:
                    sounding_bytes = self.compressed_data_bytes[sounding_byte_ind:]
                else:
                    sounding_byte_ind_2 = sounding_byte_ind_list[sounding_ind + 1]
                    sounding_bytes = self.compressed_data_bytes[sounding_byte_ind:sounding_byte_ind_2]
                    
                arr_2D = self.decompress_2D(sounding_bytes, fill_value=fill_value) #Also applies inverse transformation. 
                arr_3D[i, :, :] = arr_2D[:, :]
                
            return arr_3D
        
        
        
        elif compression_mode == 2 or compression_mode == 4:
            
            orig_dim = struct.unpack('I', self.compressed_data_bytes[i_byte:i_byte + num_bytes_I])[0]
            i_byte = i_byte + num_bytes_I
            
            num_soundings = struct.unpack('Q', self.compressed_data_bytes[i_byte:i_byte + num_bytes_Q])[0]
            i_byte = i_byte + num_bytes_Q
            
            abs_error = struct.unpack('d', self.compressed_data_bytes[i_byte:i_byte + num_bytes_d])[0]
            i_byte = i_byte + num_bytes_d
            
            T_left = np.zeros((orig_dim, orig_dim,))
            T_right = np.zeros((orig_dim, orig_dim,))
            T_left_row_sets = np.zeros((orig_dim, orig_dim,))
            T_right_col_sets = np.zeros((orig_dim, orig_dim,)) 
            filtered_mean_mat = np.zeros((orig_dim, orig_dim,))
            q_divisor_mat = np.zeros((orig_dim, orig_dim,))
            num_r_mat = np.zeros((orig_dim, orig_dim,))
            
            for i in range(orig_dim):
                for j in range(orig_dim):
                    T_left[i, j] = struct.unpack('d', self.compressed_data_bytes[i_byte:i_byte + num_bytes_d])[0]
                    i_byte = i_byte + num_bytes_d
            
            T_right[:, :] = T_left.transpose()[:, :]
            
            num_bytes_origdimsqbits = int(np.ceil(orig_dim*orig_dim / 8))
            
            T_left_row_sets_bytes = bytes(self.compressed_data_bytes[i_byte:i_byte + num_bytes_origdimsqbits])
            T_left_row_sets_bits = bitarray(endian='big')
            T_left_row_sets_bits.frombytes(T_left_row_sets_bytes)
            T_left_row_sets = np.zeros((orig_dim, orig_dim,))
            i_bit_temp = 0
            for i in range(orig_dim):
                for j in range(orig_dim):
                    if T_left_row_sets_bits[i_bit_temp]:
                        T_left_row_sets[i, j] = 1
                    i_bit_temp = i_bit_temp + 1 
            i_byte = i_byte + num_bytes_origdimsqbits
            
            T_right_col_sets[:, :] = T_left_row_sets[:, :]
            
            for i in range(orig_dim):
                for j in range(i, orig_dim):
                    filtered_mean_mat[i, j] = struct.unpack('d', self.compressed_data_bytes[i_byte:i_byte + num_bytes_d])[0]
                    if j > i:
                        filtered_mean_mat[j, i] = filtered_mean_mat[i, j]
                    i_byte = i_byte + num_bytes_d
            
            for i in range(orig_dim):
                for j in range(i, orig_dim):
                    q_divisor_mat[i, j] = struct.unpack('d', self.compressed_data_bytes[i_byte:i_byte + num_bytes_d])[0]
                    if j > i:
                        q_divisor_mat[j, i] = q_divisor_mat[i, j]
                    i_byte = i_byte + num_bytes_d
            
            for i in range(orig_dim):
                for j in range(i, orig_dim):
                    num_r_mat[i, j] = struct.unpack('Q', self.compressed_data_bytes[i_byte:i_byte + num_bytes_Q])[0]
                    if j > i:
                        num_r_mat[j, i] = num_r_mat[i, j]
                    i_byte = i_byte + num_bytes_d
            
            sounding_byte_ind_list = []
            for i in range(num_soundings):
                sounding_byte_ind = struct.unpack('Q', self.compressed_data_bytes[i_byte:i_byte + num_bytes_Q])[0]
                sounding_byte_ind_list.append(sounding_byte_ind)
                i_byte = i_byte + num_bytes_Q 
            
            
            #Invert transformation matrices: 
            T_left_inv_list, T_right_inv_list = self.invert_transformation_matrices(T_left, T_right, T_left_row_sets, T_right_col_sets) 
            
            self.compression_mode = compression_mode
            self.orig_dim = orig_dim
            self.num_soundings = num_soundings
            self.abs_error = abs_error
            self.T_left = T_left
            self.T_right = T_right
            self.T_left_row_sets = T_left_row_sets
            self.T_right_col_sets = T_right_col_sets
            self.T_left_inv_list = T_left_inv_list
            self.T_right_inv_list = T_right_inv_list 
            self.filtered_mean_mat = filtered_mean_mat
            self.q_divisor_mat = q_divisor_mat
            self.num_r_mat = num_r_mat
            self.sounding_byte_ind_list = sounding_byte_ind_list 
            
            if sounding_indices==None:
                sounding_indices = [i for i in range(num_soundings)]
            
            arr_3D = np.zeros((len(sounding_indices), orig_dim, orig_dim,))
            
            for i in tqdm(range(len(sounding_indices))): 
                sounding_ind = int(sounding_indices[i])
                sounding_byte_ind = sounding_byte_ind_list[sounding_ind]
                if sounding_ind == num_soundings - 1:
                    sounding_bytes = self.compressed_data_bytes[sounding_byte_ind:]
                else:
                    sounding_byte_ind_2 = sounding_byte_ind_list[sounding_ind + 1]
                    sounding_bytes = self.compressed_data_bytes[sounding_byte_ind:sounding_byte_ind_2]
                
                arr_2D = self.decompress_2D(sounding_bytes, fill_value=fill_value) #Also applies inverse transformation. 
                arr_3D[i, :, :] = arr_2D[:, :]
                
            return arr_3D
            
        
        else:
            print("Unknown compression mode.")
            return
        