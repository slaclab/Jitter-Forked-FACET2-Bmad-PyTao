Using the Impact setup from https://github.com/ericcropp/Impact-T_Examples/blob/main/FACET-II_Impact_Bmad/Impact_Bmad.ipynb

Changes:
* Updated the distgen.yaml t_dist to a superposition from Gaussian. Two equal duration Gaussians separated by 9 ps, 0.75:0.25 split
* Adjusted L0A phase by -10 deg
* Changed solenoid "t" (maybe kG-m? But, if so, this new value is extremely low. Frankly, nothing about this makes a lot of sense though assuming that initial number was any good.) from -0.4185 (witness ~= 8.3 um-rad, ensemble ~= 20 um-rad) to -0.1 (witness ~= 4 um-rad, ensemble ~= 9.6 um-rad), prioritizing witness

Quick and dirty; don't expect this to be a long term reference beam.

Known problems
* Uses same VCC image for both arms
* L0A phase not validated; known that the L0AFEND LPS does not match GPT reference
* Emittance, even when optimized, is higher than it should be
 * With L0A phase adjusted by -10 deg, emittance is bad (can't get witness <6 um-rad or ensemble <15 um-rad)
 * ~Same results with L0A left at 0 deg offset



