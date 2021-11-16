import os
DataDeepFieldsDir=os.environ["FINDCLUSTER_DATA_DIR"]
DicoDataNames_EN1={"PhotoCat":"%s/EN1/EN1_opt_spitzer_merged_vac_opt3as_irac4as_all_hpx_public.fits"%DataDeepFieldsDir,
                   "PzCat":"%s/EN1/EN1_opt_spitzer_merged_vac_opt3as_irac4as_all_public_pz.hdf"%DataDeepFieldsDir,
                   "MaskImage":"%s/EN1/optical_images/iband/EL_EN1_iband.fits.mask.fits"%DataDeepFieldsDir,
                   "PhysCat":"%s/EN1/en1_test_all_zp_terr.fits.cat"%DataDeepFieldsDir,
                   "PickleSave":"%s/EN1/EN1.DicoData"%DataDeepFieldsDir,
                   "RGBNames":["%s/EN1/optical_images/sw2band/EL_EN1_sw2band.fits"%DataDeepFieldsDir,
                               "%s/EN1/optical_images/Kband/EL_EN1_Kband.fits"%DataDeepFieldsDir,
                               "%s/EN1/optical_images/iband/EL_EN1_iband.fits"%DataDeepFieldsDir]}

