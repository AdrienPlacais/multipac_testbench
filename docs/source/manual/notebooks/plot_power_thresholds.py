#!/usr/bin/env python
# coding: utf-8

# # Plot power thresholds

#

# Generic libraries:

# In[1]:


from functools import partial
from pathlib import Path
import tomllib

import numpy as np


# Other libraries required for this notebook:

# In[2]:


from multipac_testbench.src.test_campaign import TestCampaign
import multipac_testbench.src.instruments as ins
from multipac_testbench.src.util.post_treaters import running_mean
from multipac_testbench.src.util.multipactor_detectors import \
    quantity_is_above_threshold


# Define the project path, load the configuration.

# In[3]:


project = Path("../data/campaign")
config_path = Path(project, "testbench_configuration.toml")

with open(config_path, "rb") as f:
    config = tomllib.load(f)


# In[4]:


frequencies = (120., 120., 120., 120., 120.)
swrs = (1., 2., 4., 10., np.inf)
info = ('', '', '', '', 'short-circuit')
filepaths = (
    Path(project, "120MHz_travelling.csv"),
    Path(project, "120MHz_SWR2.csv"),
    Path(project, "120MHz_SWR4.csv"),
    Path(project, "120MHz_SWR10.csv"),
    Path(project, "120MHz_short-circuit.csv")
)
test_campaign = TestCampaign.from_filepaths(filepaths,
                                            frequencies,
                                            swrs,
                                            config,
                                            info=info,
                                            sep='\t')


# Prepare visualisation

# In[5]:


ignored_pick_ups = ('E1', 'V1', 'V2', 'V3')
exclude = "NI9205_E1",
to_plot = (ins.CurrentProbe, ins.Power)
figsize = (20, 9)


# Smooth the current data

# In[6]:


current_smoother = partial(
    running_mean,
    n_mean=10,
    mode='same',
)

test_campaign.add_post_treater(
    current_smoother,
    ins.CurrentProbe,
)


# Set a multipactor detection criterion:

# In[7]:


current_multipactor_criterions = {'threshold': 12.,
                                  'consecutive_criterion': 10,
                                  'minimum_number_of_points': 7}
current_multipac_detector = partial(quantity_is_above_threshold,
                                    **current_multipactor_criterions)
current_multipactor_bands = test_campaign.detect_multipactor(
    current_multipac_detector,
    ins.CurrentProbe,
)


# Check that detected multipactor zones are consistent:

# In[8]:


axe = test_campaign.check_somersalo_scaling_law(
    current_multipactor_bands,
    # measurement_points_to_exclude=ignored_pick_ups,
    figsize=figsize,
    drop_idx=[-1],
    use_theoretical_r=False,
)


# In[ ]:
