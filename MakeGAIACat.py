from astroquery.gaia import Gaia



tables = Gaia.load_tables(only_names=True)
ra=15.*(16+10./60+55.650/3600)
dec=54+58/60.+18.36/3600.
R=1.64274
job = Gaia.launch_job_async("SELECT * \
FROM gaiadr1.gaia_source \
WHERE CONTAINS(POINT('ICRS',gaiadr1.gaia_source.ra,gaiadr1.gaia_source.dec),CIRCLE('ICRS',%f,%f,%f))=1 \
AND phot_g_mean_mag < 10;"%(ra,dec,R) , dump_to_file=True)

job = Gaia.launch_job_async("SELECT * \
FROM gaiadr1.gaia_source \
WHERE CONTAINS(POINT('ICRS',gaiadr1.gaia_source.ra,gaiadr1.gaia_source.dec),CIRCLE('ICRS',%f,%f,%f))=1;"%(ra,dec,R) , dump_to_file=True)

job = Gaia.launch_job_async("SELECT * \
FROM gaiadr1.gaia_source \
WHERE CONTAINS(POINT('ICRS',gaiadr1.gaia_source.ra,gaiadr1.gaia_source.dec),CIRCLE('ICRS',56.75,24.1167,2))=1;" \
, dump_to_file=True)


return job
