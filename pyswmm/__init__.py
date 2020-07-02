# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2014 Bryant E. McDonnell
#
# Licensed under the terms of the BSD2 License
# See LICENSE.txt for details
# -----------------------------------------------------------------------------
"""Python Wrapper for Stormwater Management Model (SWMM5)."""

# Local imports
from pyswmm.links import Link, Links
from pyswmm.lidcontrols import LidControls, LidControl
from pyswmm.lidgroups import LidGroups, LidGroup, LidUnit
from pyswmm.nodes import Node, Nodes
from pyswmm.simulation import Simulation
from pyswmm.subcatchments import Subcatchment, Subcatchments
from pyswmm.system import SystemStats
from pyswmm.raingages import RainGages, RainGage

VERSION_INFO = (0, 6, 0)
__version__ = '.'.join(map(str, VERSION_INFO))
__swmm_version__ = '5.2.1'
__swmm_sha__ = '854df626b32f74e5cfb3a07fbea3f51b5d7855bb'
__epaswmm_version_basis__ = '5.1.14.dev0'
__author__ = 'Bryant E. McDonnell (EmNet LLC) - bemcdonnell@gmail.com'
__copyright__ = 'Copyright (c) 2016 Bryant E. McDonnell (See AUTHORS)'
__licence__ = 'BSD2'
__all__ = [
    Link, Links, LidControls, LidGroups, Node, Nodes, Subcatchment, Subcatchments, Simulation,
    SystemStats, RainGages, RainGage
]