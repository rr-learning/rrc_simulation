#!/bin/bash
# Update the local robot_properties_fingers files

rrc_simulation=$(rospack find rrc_simulation)
robot_properties_fingers=$(rospack find robot_properties_fingers)
local_properties=${rrc_simulation}/python/rrc_simulation/robot_properties_fingers

rm -rf ${local_properties}/*
cp -r ${robot_properties_fingers}/meshes ${local_properties}
cp -r ${robot_properties_fingers}/urdf ${local_properties}
