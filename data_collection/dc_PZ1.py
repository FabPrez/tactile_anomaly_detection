import dc_pipeline

pezzo_num = 1
number_of_diff_poses = 4
number_of_acquisition_per_pose = 1
fault_data_acquisiton = True

if __name__ == "__main__":
    dc_pipeline.start_pipeline(pezzo_num,number_of_diff_poses,number_of_acquisition_per_pose,fault_data_acquisiton)