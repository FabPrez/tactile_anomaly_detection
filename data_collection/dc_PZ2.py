import dc_pipeline

pezzo_num = 2
number_of_diff_poses = 9
number_of_acquisition_per_pose = 10
fault_data_acquisiton = False


if __name__ == "__main__":
    dc_pipeline.start_pipeline(pezzo_num,number_of_diff_poses,number_of_acquisition_per_pose,fault_data_acquisiton)