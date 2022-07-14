# CMake generated Testfile for 
# Source directory: /workspace/nifty_reg_source/reg-test
# Build directory: /workspace/nifty_reg_build/reg-test
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(interpolation_3D_NN "reg_test_interp" "/workspace/nifty_reg_source/reg-test/reg-test-data/brainweb_3D.nii.gz" "/workspace/nifty_reg_source/reg-test/reg-test-data/test_DEF_BW_3D.nii.gz" "0" "/workspace/nifty_reg_source/reg-test/reg-test-data/test_warped_BW_3D_NN.nii.gz")
add_test(interpolation_3D_LIN "reg_test_interp" "/workspace/nifty_reg_source/reg-test/reg-test-data/brainweb_3D.nii.gz" "/workspace/nifty_reg_source/reg-test/reg-test-data/test_DEF_BW_3D.nii.gz" "1" "/workspace/nifty_reg_source/reg-test/reg-test-data/test_warped_BW_3D_LIN.nii.gz")
add_test(interpolation_3D_SPL "reg_test_interp" "/workspace/nifty_reg_source/reg-test/reg-test-data/brainweb_3D.nii.gz" "/workspace/nifty_reg_source/reg-test/reg-test-data/test_DEF_BW_3D.nii.gz" "3" "/workspace/nifty_reg_source/reg-test/reg-test-data/test_warped_BW_3D_SPL.nii.gz")
add_test(interpolation_2D_NN "reg_test_interp" "/workspace/nifty_reg_source/reg-test/reg-test-data/brainweb_2D.nii.gz" "/workspace/nifty_reg_source/reg-test/reg-test-data/test_DEF_BW_2D.nii.gz" "0" "/workspace/nifty_reg_source/reg-test/reg-test-data/test_warped_BW_2D_NN.nii.gz")
add_test(interpolation_2D_LIN "reg_test_interp" "/workspace/nifty_reg_source/reg-test/reg-test-data/brainweb_2D.nii.gz" "/workspace/nifty_reg_source/reg-test/reg-test-data/test_DEF_BW_2D.nii.gz" "1" "/workspace/nifty_reg_source/reg-test/reg-test-data/test_warped_BW_2D_LIN.nii.gz")
add_test(interpolation_2D_SPL "reg_test_interp" "/workspace/nifty_reg_source/reg-test/reg-test-data/brainweb_2D.nii.gz" "/workspace/nifty_reg_source/reg-test/reg-test-data/test_DEF_BW_2D.nii.gz" "3" "/workspace/nifty_reg_source/reg-test/reg-test-data/test_warped_BW_2D_SPL.nii.gz")
add_test(mat44_operations "reg_test_mat44_operations")
